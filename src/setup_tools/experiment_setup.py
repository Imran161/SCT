# класс для создания эксперимента

from typing import Any, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader

from ..losses.losses import SMOOTH
from ..losses.losses import (
    bce,
    global_focus_loss,
    strong_combined_loss,
    weak_combined_loss,
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
        if not use_class_weight:
            all_class_weights = None
            self.use_cls = "off"
        else:
            count_data = self.batch_size * len(self.train_loader)
            pos_weights = count_data / (2 * self.TotalTrain + SMOOTH)  # ML
            neg_weights = count_data / (2 * (count_data - self.TotalTrain + SMOOTH))
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

        if not use_pixel_opt:
            self.use_pixel_opt = "off"

        if not use_pixel_weight:
            pixel_all_class_weights = None
            self.use_pixel = "off"
        else:
            for batch_idx, train_batch in enumerate(self.train_loader):
                images = train_batch["images"]
                batch_size, channels, height, width = images.shape

                print(f"Batch size: {batch_size}")
                print(f"Number of channels: {channels}")
                print(f"Height: {height}")
                print(f"Width: {width}")
                break

            pixel_count_data = (
                self.batch_size * len(self.train_loader) * height * width
            )  # 256 * 256
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

        experiment_name = f"{power}_loss_clsW_{self.use_cls}_pixW_{self.use_pixel}_pixOpt_{self.use_pixel_opt}"
        # experiment_name = f"{power}_loss_class_weights_{self.use_cls}_pixel_weights_{self.use_pixel}_pixel_opt_{self.use_pixel_opt}"

        if "weak_loss" in experiment_name:
            criterion = weak_combined_loss
        elif "strong_loss" in experiment_name:
            criterion = strong_combined_loss
        elif "focus_loss" in experiment_name:
            criterion = global_focus_loss
        elif "bce_loss" in experiment_name:
            criterion = bce
        else:
            raise ValueError("Invalid experiment name")

        return all_class_weights, pixel_all_class_weights, experiment_name, criterion
