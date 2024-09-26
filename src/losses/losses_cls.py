import torch

SMOOTH = 1e-8  # вот это потом можно в отдельный файл с константами перенести


class BCEMeanLoss:
    @staticmethod
    def forward(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = -target * torch.log(input + SMOOTH) - (1 - target) * torch.log(
            1 - input + SMOOTH
        )
        return loss.mean()


class BCELoss:
    @staticmethod
    def forward(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = -target * torch.log(input + SMOOTH) - (1 - target) * torch.log(
            1 - input + SMOOTH
        )
        return loss


class FocalLoss:
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor = None,
        reduction: str = "mean",
        normalized: bool = False,
        reduced_threshold=None,
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.normalized = normalized
        self.reduced_threshold = reduced_threshold

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        size = target.shape
        target = target.type(input.type())

        loss_ce = BCELoss.forward(input, target)

        pt = torch.exp(-loss_ce)

        if self.reduced_threshold is None:
            focal_term = (1.0 - pt).pow(self.gamma)
        else:
            focal_term = ((1.0 - pt) / self.reduced_threshold).pow(self.gamma)
            focal_term[pt < self.reduced_threshold] = 1

        loss_focal = focal_term * loss_ce

        if self.alpha is not None:
            for i in range(size[0]):
                for j in range(size[1]):
                    weight_matrix = (
                        (target[i, j]) * self.alpha[0][j]
                        + (1 - target[i, j]) * self.alpha[1][j]
                    )
                    loss_focal[i, j] = loss_focal[i, j] * weight_matrix

        if self.reduction == "mean":
            loss_focal = loss_focal.mean()
        elif self.reduction == "sum":
            loss_focal = loss_focal.sum()
        elif self.reduction == "batchwise_mean":
            loss_focal = loss_focal.sum(0)

        return loss_focal


class WeakIoULoss:
    def __init__(self, class_weight=None):
        self.class_weight = class_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        size = target.shape

        intersection = (input * target).float().sum((-1, -2))
        union = ((input + target).float().sum((-1, -2)) - intersection).sum((-1, -2))
        iou = 1 - (intersection) / (union + SMOOTH)

        for i in range(size[0]):
            for j in range(size[1]):
                if target[i, j].max() == 0:
                    iou[i, j] *= 0
                else:
                    if self.class_weight is not None:
                        iou[i, j] *= self.class_weight[0][j]

        return iou.mean()


class StrongIoULoss:
    def __init__(self, class_weight=None):
        self.class_weight = class_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        size = target.shape

        intersection = (input * target).float().sum((-1, -2))
        union = ((input + target).float().sum((-1, -2)) - intersection).sum((-1, -2))
        iou = 1 - (intersection) / (union + SMOOTH)

        count = 0
        for i in range(size[0]):
            for j in range(size[1]):
                if target[i, j].max() == 1:
                    count += 1
                    if self.class_weight is not None:
                        iou[i, j] *= self.class_weight[2][j]
                else:
                    iou[i, j] *= 0

        if count == 0:
            return iou.sum() * 0
        return iou.sum() / count


class WeakCombinedLoss:
    def __init__(self, class_weight, alpha):
        self.focal_loss = FocalLoss(alpha=alpha)
        self.weak_iou_loss = WeakIoULoss(class_weight)

    def forward(self, input, target):
        loss1 = self.focal_loss.forward(input, target)
        loss2 = self.weak_iou_loss.forward(input, target)
        return (loss1 + loss2) / 2


class StrongCombinedLoss:
    def __init__(self, class_weight, alpha):
        self.focal_loss = FocalLoss(alpha=alpha)
        self.strong_iou_loss = StrongIoULoss(class_weight)

    def forward(self, input, target):
        loss1 = self.focal_loss.forward(input, target)
        loss2 = self.strong_iou_loss.forward(input, target)
        return (loss1 + loss2) / 2


class GlobalFocusLoss:
    def __init__(self, mode="ML"):
        self.mode = mode
        self.global_loss_sum = 0.0
        self.global_loss_numel = 0

    def forward(self, input: torch.Tensor, target: torch.Tensor, train_mode=True):
        if self.mode == "ML":
            loss_bce = -(
                target * torch.log(input + SMOOTH)
                + (1 - target) * torch.log(1 - input + SMOOTH)
            )

        elif self.mode == "MC":
            loged_target = torch.log(input + SMOOTH)
            loss_bce = -target * loged_target

        if train_mode:
            self.global_loss_sum += loss_bce.sum().item()
            self.global_loss_numel += loss_bce.numel()

            pt = torch.exp(loss_bce - self.global_loss_sum / self.global_loss_numel)
            loss = loss_bce * pt
            loss_mean = torch.mean(loss)

        else:
            loss_mean = torch.mean(loss_bce)

        return loss_mean

    def reset_global_loss(self):
        """Сбросить накопленные значения потерь."""
        self.global_loss_sum = 0.0
        self.global_loss_numel = 0
