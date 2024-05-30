import torch
import numpy as np
import cv2
import os
from torch.utils.data import DataLoader
from typing import Tuple, Optional, List, Any
import random


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_direct_subdirectories(directory):
    subdirectories = [
        d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))
    ]
    return [os.path.join(directory, subdir) for subdir in subdirectories]


SCT_base_classes = [
    {"id": 1, "name": "1", "summable_masks": [1], "subtractive_masks": []},
    {"id": 2, "name": "2", "summable_masks": [2], "subtractive_masks": []},
    {"id": 3, "name": "3", "summable_masks": [3], "subtractive_masks": []},
    {"id": 4, "name": "4", "summable_masks": [4], "subtractive_masks": []},
    {"id": 5, "name": "5", "summable_masks": [5], "subtractive_masks": []},
]

# SCT_out_classes = [
#                     {'id': 1, 'name': 'insult_type_1', "summable_masks":[1], "subtractive_masks":[]},
#                     {'id': 2, 'name': 'insult_type_2', "summable_masks":[2], "subtractive_masks":[]},
#                     {'id': 3, 'name': 'insult_type_3', "summable_masks":[3], "subtractive_masks":[]},
#                     {'id': 4, 'name': 'insult_type_4', "summable_masks":[4], "subtractive_masks":[]},
#                    ]

SCT_out_classes = [
    {
        "id": 1,
        "name": "Внутримозговое кровоизлияние",
        "summable_masks": [1],
        "subtractive_masks": [],
    },
    {
        "id": 2,
        "name": "Субарахноидальное кровоизлияние",
        "summable_masks": [2],
        "subtractive_masks": [],
    },
    {
        "id": 3,
        "name": "Cубдуральное кровоизлияние,",
        "summable_masks": [3],
        "subtractive_masks": [],
    },
    {
        "id": 4,
        "name": "Эпидуральное кровоизлияние",
        "summable_masks": [4],
        "subtractive_masks": [],
    },
]


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


def iou_metric(outputs: torch.Tensor, labels: torch.Tensor, num_classes: int):
    ious = torch.zeros(num_classes)

    for class_idx in range(num_classes):
        binary_outputs = (outputs[:, class_idx, :, :] > 0.5).byte()
        binary_labels = (labels[:, class_idx, :, :]).byte()

        intersection = (binary_outputs & binary_labels).sum((-1, -2))
        union = (binary_outputs | binary_labels).sum((-1, -2))

        iou = (intersection) / (union + 1e-8)
        ious[class_idx] = iou.mean()

    return ious


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


class ImageVisualizer:
    def __init__(self, output_path):
        self.output_path = output_path

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def threshold_predictions(self, pred_masks, threshold=0.5):
        rounded_masks = pred_masks >= threshold
        return np.ceil(rounded_masks)

    def visualize(
        self, images, true_masks, pred_masks, class_names_dict, colors, epoch=None
    ):
        # print("type true_masks", type(true_masks))
        true_masks = true_masks.detach().cpu().numpy()
        # print("pred_masks type", type(pred_masks)) # <class 'torch.Tensor'>

        pred_masks = pred_masks.detach().cpu().numpy()
        pred_masks = self.threshold_predictions(pred_masks)

        for i in range(len(images)):
            image = images[i][0].cpu().numpy()
            image = (image * 255).astype(np.uint8)

            true_image_with_contours = np.zeros(
                (image.shape[0], image.shape[1], 3), dtype=np.uint8
            )
            # Копируем одноканальное изображение в канал синего цвета (Blue)
            true_image_with_contours[:, :, 0] = image
            # Копируем одноканальное изображение в канал зеленого цвета (Green)
            true_image_with_contours[:, :, 1] = image
            # Копируем одноканальное изображение в канал красного цвета (Red)
            true_image_with_contours[:, :, 2] = image

            for j in range(np.shape(true_masks[i])[0]):
                # print("np.shape(true_masks[i])[0]", np.shape(true_masks[i])[0]) # 4
                color = (colors[j][0][0], colors[j][0][1], colors[j][0][2])
                contours, _ = cv2.findContours(
                    true_masks[i][j].astype(int).astype(np.uint8),
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                true_image_with_contours = cv2.drawContours(
                    true_image_with_contours, contours, -1, color, 2
                )

                # values, counts = np.unique(true_masks[i][j], return_counts=True)
                # for v, c in zip(values, counts):
                #     print(f"v:{v}, c:{c}")

                if np.max(true_masks[i][j]) == 1:
                    #     # text = list_of_name_out_classes[j]
                    # print("i, j",i, j)
                    text = class_names_dict[j + 1]
                    # print("text", text)
                    # plt.text(2000, k, text , color = (self.colors[i][0][0]/255, self.colors[i][0][1]/255, self.colors[i][0][2]/255))
                    cv2.putText(
                        true_image_with_contours,
                        f"true class: {text}",
                        (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                # k+=50
            #     class_name = class_names_dict[j+1]
            # cv2.putText(true_image_with_contours, f"True Mask: {class_name}", (10, 20 + j * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            pred_image_with_contours = np.zeros(
                (image.shape[0], image.shape[1], 3), dtype=np.uint8
            )
            # Копируем одноканальное изображение в канал синего цвета (Blue)
            pred_image_with_contours[:, :, 0] = image
            # Копируем одноканальное изображение в канал зеленого цвета (Green)
            pred_image_with_contours[:, :, 1] = image
            # Копируем одноканальное изображение в канал красного цвета (Red)
            pred_image_with_contours[:, :, 2] = image

            for j in range(np.shape(pred_masks[i])[0]):
                color = (colors[j][0][0], colors[j][0][1], colors[j][0][2])
                contours, _ = cv2.findContours(
                    pred_masks[i][j].astype(int).astype(np.uint8),
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                # print("contours", contours)
                pred_image_with_contours = cv2.drawContours(
                    pred_image_with_contours, contours, -1, color, 2
                )

                # for class_idx, class_name in class_names_dict.items():
                #     cv2.putText(pred_image_with_contours, f"true class: {class_name}", (10, 20 * class_idx), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                if np.max(pred_masks[i][j]) == 1:
                    # text = list_of_name_out_classes[j]
                    # print("i, j",i, j)
                    text = class_names_dict[j + 1]
                    # print("text", text)
                    # plt.text(2000, k, text , color = (self.colors[i][0][0]/255, self.colors[i][0][1]/255, self.colors[i][0][2]/255))
                    cv2.putText(
                        pred_image_with_contours,
                        f"pred class: {text}",
                        (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    # k+=50
            #     class_name = class_names_dict[j+1]  # Имя класса
            # cv2.putText(pred_image_with_contours, f"Pred Mask: {class_name}", (10, 20 + j * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            combined_image = np.concatenate(
                (true_image_with_contours, pred_image_with_contours), axis=1
            )

            values, counts = np.unique(true_masks[i], return_counts=True)
            if len(values) > 1:
                if epoch is None:
                    cv2.imwrite(f"{self.output_path}/image_{i}.jpg", combined_image)
                else:
                    cv2.imwrite(
                        f"{self.output_path}/epoch_{epoch}_image_{i}.jpg",
                        combined_image,
                    )


class ExperimentSetup:
    def __init__(
        self,
        train_loader: DataLoader,
        TotalTrain: np.ndarray,
        pixel_TotalTrain: np.ndarray,
        batch_size: int,
        num_classes: int,
        use_background=False,
    ) -> None:
        self.train_loader = train_loader
        self.TotalTrain = TotalTrain
        self.pixel_TotalTrain = pixel_TotalTrain
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.use_background = use_background
        self.use_cls = "on"
        self.use_pixel = "on"
        self.use_pixel_opt = "on"

    def setup_experiment(
        self,
        use_class_weight: bool,
        use_pixel_weight: bool,
        use_pixel_opt: bool,
        power: str,
    ) -> Tuple[Optional[List[float]], Optional[List[float]], str, Any]:
        if use_class_weight == False:
            all_class_weights = None
            self.use_cls = "off"
        else:
            count_data = self.batch_size * len(self.train_loader)
            pos_weights = count_data / (2 * self.TotalTrain)  # ML
            neg_weights = count_data / (2 * (count_data - self.TotalTrain))
            class_weights = self.TotalTrain[1:].sum() / (
                self.TotalTrain * self.num_classes
            )  # MC
            class_weight_use_background = self.TotalTrain.sum() / (
                self.TotalTrain * self.num_classes + 1
            )  # это если вдруг с фоном веса нужны

            all_class_weights = [pos_weights, neg_weights]  # , class_weights]

            if self.use_background:
                all_class_weights.append(class_weight_use_background)
            else:
                all_class_weights.append(class_weights)

        if use_pixel_opt == False:
            self.use_pixel_opt = "off"

        if use_pixel_weight == False:
            pixel_all_class_weights = None
            self.use_pixel = "off"
        else:
            pixel_count_data = self.batch_size * len(self.train_loader) * 256 * 256
            pixel_pos_weights = pixel_count_data / (2 * self.pixel_TotalTrain)
            pixel_neg_weights = pixel_count_data / (
                2 * (pixel_count_data - self.pixel_TotalTrain)
            )
            # pixel_class_weights = pixel_count_data / (self.num_classes * self.pixel_TotalTrain)
            pixel_class_weights = self.pixel_TotalTrain[1:].sum() / (
                self.pixel_TotalTrain * self.num_classes
            )
            pixel_class_weight_use_background = self.pixel_TotalTrain.sum() / (
                self.pixel_TotalTrain * self.num_classes + 1
            )

            pixel_all_class_weights = [
                pixel_pos_weights,
                pixel_neg_weights,
            ]  # , pixel_class_weights]

            if self.use_background:
                pixel_all_class_weights.append(pixel_class_weight_use_background)
            else:
                pixel_all_class_weights.append(pixel_class_weights)

        experiment_name = f"{power}_loss_class_weights_{self.use_cls}_pixel_weights_{self.use_pixel}_pixel_opt_{self.use_pixel_opt}"

        if "weak_loss" in experiment_name:
            criterion = weak_combined_loss
        elif "strong_loss" in experiment_name:
            criterion = strong_combined_loss
        else:
            raise ValueError("Invalid experiment name")

        return all_class_weights, pixel_all_class_weights, experiment_name, criterion
