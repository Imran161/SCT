# сохранение метрик, чтобы из функции обучения их убрать

import os
import torch
from torch.utils.tensorboard import SummaryWriter
from ..utils.utils import save_best_metrics_to_csv


class MetricsSaver:
    def __init__(self, experiment_name, num_classes, class_names_dict):
        self.experiment_name = experiment_name
        self.num_classes = num_classes
        self.class_names_dict = class_names_dict
        self.writer = SummaryWriter(log_dir=f"runs_kidneys/{experiment_name}_logs")
        self.best_loss = float("inf")
        self.best_metrics = None

    def save_metrics(
        self,
        epoch,
        train_metrics,
        val_metrics,
        train_loss_avg,
        val_loss_avg,
        train_iou_avg,
        val_iou_avg,
        optimizer,
        model,
    ):
        # Сохранение метрик трейна
        self._save_individual_metrics(epoch, train_metrics, "Train")
        self.writer.add_scalar("Loss/train", train_loss_avg, epoch)
        for class_idx, iou_value in enumerate(train_iou_avg):
            class_name = self.class_names_dict[class_idx + 1]
            self.writer.add_scalar(f"My_train_IoU/{class_name}", iou_value, epoch)

        # Сохранение метрик валидации
        self._save_individual_metrics(epoch, val_metrics, "Val")
        self.writer.add_scalar("Loss/validation", val_loss_avg, epoch)
        for class_idx, iou_value in enumerate(val_iou_avg):
            class_name = self.class_names_dict[class_idx + 1]
            self.writer.add_scalar(f"My_val_IoU/{class_name}", iou_value, epoch)

        # Сохранение lr
        self.writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

        # Сохранение лучшей модели и метрик
        if val_loss_avg < self.best_loss:
            self.best_loss = val_loss_avg
            self.best_metrics = {
                "experiment": self.experiment_name.split("_")[0],
                "epoch": epoch,
                "train_loss": train_loss_avg,
                "val_loss": val_loss_avg,
                "val_metrics": {
                    "IOU": val_metrics["IOU"],
                    "F1": val_metrics["F1"],
                    "area_probs_F1": val_metrics["area_probs_F1"],
                },
            }
            self._save_best_model_and_metrics(model, "best", epoch)

    def _save_individual_metrics(self, epoch, metrics, mode):
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if len(value.size()) > 0:
                    # средняя метрика по классам
                    self.writer.add_scalar(
                        f"{mode}/Mean/{key}", value.mean().item(), epoch
                    )
                    for i, val in enumerate(value):
                        class_name = self.class_names_dict[i + 1]
                        self.writer.add_scalar(
                            f"{mode}/{key}/{class_name}", val.item(), epoch
                        )
                else:
                    self.writer.add_scalar(f"{mode}/{key}", value.item(), epoch)

    def _save_best_model_and_metrics(self, model, model_type, epoch):
        model_path = f"kidneys_{model_type}_models"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(
            model.state_dict(),
            f"{model_path}/{model_type}_{self.experiment_name}_model.pth",
        )

        csv_file = f"{model_path}/{model_type}_metrics.csv"
        save_best_metrics_to_csv(self.best_metrics, csv_file)

    def close_writer(self):
        self.writer.close()
