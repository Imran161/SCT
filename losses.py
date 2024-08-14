import torch
import torch.nn.functional as F


def binary_cross_entropy(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    loss = -labels * torch.log(outputs + 0.00001) - (1 - labels) * torch.log(
        1 - outputs + 0.00001
    )
    return loss


def focal_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    # alpha = 1,
    alpha: torch.Tensor = None,
    # pixel_weight: bool = True,
    reduction: str = "mean",
    normalized: bool = False,
    reduced_threshold=None,
    eps: float = 1e-4,
) -> torch.Tensor:
    size = target.shape
    # print("size", size) # torch.Size([16, 4, 256, 256])
    target = target.type(output.type())

    loss_ce = binary_cross_entropy(output, target)

    # веса для focal loss
    pt = torch.exp(-loss_ce)
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1

    loss_focal = focal_term * loss_ce

    if alpha is not None:
        for i in range(size[0]):
            for j in range(size[1]):
                weight_matrix = (
                    (target[i, j]) * alpha[0][j] + (1 - target[i, j]) * alpha[1][j]
                )

                loss_focal[i, j] = loss_focal[i, j] * weight_matrix

    # Но ваще это нужно
    # if alpha is not None:
    #     loss_focal *= alpha * target + (1 - alpha) * (1 - target)

    # if normalized:
    #     norm_factor = focal_term.sum().clamp_min(eps)
    #     loss_focal /= norm_factor

    if reduction == "mean":
        loss_focal = loss_focal.mean()
    if reduction == "sum":
        loss_focal = loss_focal.sum()
    if reduction == "batchwise_mean":
        loss_focal = loss_focal.sum(0)

    return loss_focal


def weak_iou_loss(outputs: torch.Tensor, labels: torch.Tensor, class_weight=None):
    SMOOTH = 1e-8
    size = labels.shape

    intersection = (outputs * labels).float().sum((-1, -2))
    union = ((outputs + labels).float().sum((-1, -2)) - intersection).sum((-1, -2))
    iou = 1 - (intersection) / (union + SMOOTH)

    for i in range(size[0]):
        for j in range(size[1]):
            if labels[i, j].max() == 0:
                iou[i, j] *= 0
            else:
                if class_weight is not None:
                    iou[i, j] *= class_weight[0][j]  # [0][j] положительный вес класса j

    return iou.mean()


def strong_iou_loss(outputs: torch.Tensor, labels: torch.Tensor, class_weight=None):
    SMOOTH = 1e-8
    size = labels.shape

    intersection = (outputs * labels).float().sum((-1, -2))
    union = ((outputs + labels).float().sum((-1, -2)) - intersection).sum((-1, -2))
    iou = 1 - (intersection) / (union + SMOOTH)

    count = 0
    for i in range(size[0]):
        for j in range(size[1]):
            if labels[i, j].max() == 1:
                count += 1
                if class_weight is not None:
                    iou[i, j] *= class_weight[2][j]

            else:
                iou[i, j] *= 0

    if count == 0:
        return iou.sum() * 0
    # else:
    return iou.sum() / count


def weak_combined_loss(output, target, class_weight, alpha):
    loss1 = focal_loss(
        output,
        target,
        gamma=2.0,
        #    alpha=1,
        alpha=alpha,
        #    pixel_weight = pixel_weight,
        reduction="mean",
        normalized=False,
        reduced_threshold=None,
        eps=1e-4,
    )

    loss2 = weak_iou_loss(output, target, class_weight)
    return (loss1 + loss2) / 2

    # return loss1


def strong_combined_loss(output, target, class_weight, alpha):
    loss1 = focal_loss(
        output,
        target,
        gamma=2.0,
        #    alpha=1,
        alpha=alpha,
        #    pixel_weight = pixel_weight,
        reduction="mean",
        normalized=False,
        reduced_threshold=None,
        eps=1e-4,
    )

    loss2 = strong_iou_loss(output, target, class_weight)
    return (loss1 + loss2) / 2


def global_focus_loss(label, true_label, mode="ML", global_loss_sum=0, global_loss_numel=0):

    smooth=0.00001
   
    if mode == "ML":
        label = F.sigmoid(label)
        label = label+smooth
        loss = -(true_label * torch.log(label) + (1 - true_label) * torch.log(1 - label))
        
    if mode == "MC":
        label = F.softmax(label)
        label = label+smooth
        logged_label = torch.log(label)
        loss = - true_label * logged_label
    
    global_loss_sum += loss.sum().item()
    global_loss_numel += loss.numel()
    
    pt = torch.exp(loss - global_loss_sum/global_loss_numel)
    loss = loss*pt
    loss_mean = torch.mean(loss)

    return loss_mean

сравнить на почках ее с Надо bce, focal без всего, focal с пиксельными с FPN а потом linknet
новые данные нарезать скинуть Саше в трех проекциях и запустить с kidneys_pat_out_classesс FPN а потом с linknet, а дальше новые лоссы будем использовать