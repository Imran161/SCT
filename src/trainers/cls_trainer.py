import torch
import warnings
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import UndefinedMetricWarning

from metrics import Detection_metrics
from utils import (
    SCT_base_classes,
    SCT_out_classes,
    binary_cross_entropy,
    focal_loss,
    iou_metric,
    weak_iou_loss,
    strong_iou_loss,
    weak_combined_loss,
    strong_combined_loss,
    ImageVisualizer,
    ExperimentSetup,
)

from utils import set_seed

set_seed(64)


class ModelTrainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        num_epochs,
        train_loader,
        val_loader,
        device,
        num_classes,
        experiment_name,
        all_class_weights=None,
        alpha=None,
        use_opt_pixel_weight=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        self.experiment_name = experiment_name
        self.all_class_weights = all_class_weights
        self.alpha = alpha
        self.use_opt_pixel_weight = use_opt_pixel_weight

        self.train_predict_path = (
            f"/home/imran-nasyrov/sct_project/sct_data/predict_test/train"
        )
        self.train_image_visualizer = ImageVisualizer(self.train_predict_path)

        self.val_predict_path = (
            f"/home/imran-nasyrov/sct_project/sct_data/predict_test/val"
        )
        self.val_image_visualizer = ImageVisualizer(self.val_predict_path)

        self.writer = SummaryWriter(log_dir=f"runs/{self.experiment_name}_logs")
        self.metrics_calculator = Detection_metrics(
            mode="ML", num_classes=self.num_classes
        )

        self.class_names_dict = {
            class_info["id"]: class_info["name"] for class_info in SCT_out_classes
        }
        self.classes = list(self.class_names_dict.keys())
        self.weight_opt = Weight_opt_class(criterion, classes, None)

        self.best_loss = 100

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss_sum = 0.0
            train_iou_sum = torch.zeros(self.num_classes)

            with tqdm(
                total=len(self.train_loader),
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                unit="batch",
            ) as pbar:
                for train_batch in self.train_loader:
                    self.optimizer.zero_grad()
                    images = train_batch["images"].to(self.device)
                    masks = train_batch["masks"][:, 1:, :, :].to(self.device)

                    outputs = self.model(images)
                    outputs = torch.sigmoid(outputs)

                    loss = self.criterion(
                        outputs,
                        masks,
                        [x[1:] for x in self.all_class_weights],
                        self.alpha,
                    )
                    loss.backward()
                    self.optimizer.step()

                    train_loss_sum += loss.item()
                    train_iou_batch = iou_metric(outputs, masks, self.num_classes)
                    train_iou_sum += train_iou_batch

                    metrics_calculator.update_counter(masks, outputs)

                    n += 1
                    pbar.set_postfix(loss=train_loss_sum / n)
                    pbar.update(1)

            train_loss_avg = train_loss_sum / len(self.train_loader)
            train_iou_avg = train_iou_sum / len(self.train_loader)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                train_metrics = metrics_calculator.calculate_metrics()

            for key, value in train_metrics.items():
                if isinstance(value, torch.Tensor):
                    if len(value.size()) > 0:
                        writer.add_scalar(
                            f"Train/Mean/{key}", value.mean().item(), epoch
                        )
                        for i, val in enumerate(value):
                            writer.add_scalar(
                                f"Train/{key}/Class_{i}", val.item(), epoch
                            )
                    else:
                        writer.add_scalar(f"Train/{key}", value.item(), epoch)

            print(
                f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss_avg}, Train IoU: {train_iou_avg}"
            )

    def validate(self):
        val_predict_path = f"/home/imran-nasyrov/sct_project/sct_data/predict_test/val"
        val_image_visualizer = ImageVisualizer(val_predict_path)

        writer = SummaryWriter(log_dir=f"runs/{self.experiment_name}_logs")
        metrics_calculator = Detection_metrics(mode="ML", num_classes=self.num_classes)

        class_names_dict = {
            class_info["id"]: class_info["name"] for class_info in SCT_out_classes
        }

        best_loss = 100

        for epoch in range(self.num_epochs):
            self.model.eval()
            val_loss_sum = 0.0
            val_iou_sum = torch.zeros(self.num_classes)

            with torch.no_grad():
                for val_batch in self.val_loader:
                    images_val = val_batch["images"].to(self.device)
                    masks_val = val_batch["masks"][:, 1:].to(self.device)
                    outputs_val = self.model(images_val)

                    outputs_val = torch.sigmoid(outputs_val)
                    val_loss_sum += self.criterion(
                        outputs_val, masks_val, None, None
                    ).item()

                    val_iou_batch = iou_metric(outputs_val, masks_val, self.num_classes)
                    val_iou_sum += val_iou_batch

                    metrics_calculator.update_counter(masks_val, outputs_val)

            val_loss_avg = val_loss_sum / len(self.val_loader)
            val_iou_avg = val_iou_sum / len(self.val_loader)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                metrics = metrics_calculator.calculate_metrics()

            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    if len(value.size()) > 0:
                        writer.add_scalar(f"Val/Mean/{key}", value.mean().item(), epoch)
                        for i, val in enumerate(value):
                            writer.add_scalar(f"Val/{key}/Class_{i}", val.item(), epoch)
                    else:
                        writer.add_scalar(f"Val/{key}", value.item(), epoch)

            for class_idx, iou_value in enumerate(val_iou_avg):
                class_name = class_names_dict[class_idx + 1]
                writer.add_scalar(f"My_val_IoU/{class_name}", iou_value, epoch)

            writer.add_scalar("Loss/validation", val_loss_avg, epoch)
            print(
                f"Epoch [{epoch+1}/{self.num_epochs}], Val Loss: {val_loss_avg},  Val IoU: {val_iou_avg}"
            )

        writer.close()
