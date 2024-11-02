import os
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, auc
from torchvision import transforms

import warnings
from abc import ABC, abstractmethod
import os
import time
import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from sklearn.exceptions import UndefinedMetricWarning
import segmentation_models_pytorch as smp

from ..losses.losses_cls import (
    WeakCombinedLoss,
    FocalLoss,
)  # Подключи кастомные функции лосса
from ..datamanager.coco_dataloaders import SINUSITE_COCODataLoader
from ..metrics.metrics import DetectionMetrics

from ..datamanager.coco_classes import (
    kidneys_base_classes,
    kidneys_pat_out_classes,
)
#######################################################
# хотел так сделать но сделаю как ниже


class ExperimentSetup:
    def __init__(self, config):
        self.config = config
        self.exp_name = self.create_experiment_dir()

    def create_experiment_dir(self):
        timestamp = time.strftime("%Y-%m-%d__%H_%M_%S")
        exp_name = f"{timestamp}_{self.config['exp_name']}"
        os.makedirs(f"{self.config['runs_path']}/{exp_name}/models", exist_ok=True)
        return exp_name


class SegTrainer:
    def __init__(self, net, train_dataloader, val_dataloader, config):
        self.net = net
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.experiment = ExperimentSetup(config)

        # Переменные и функции
        self.vars = {
            "phase": None,
            "epoch": 0,
            "actual_epoch_loss": 0.0,
            "actual_num_of_step": 0.0,
            "best_epoch_loss": float("inf"),
            "counter": config["early_stop"],
        }

        self.loss_functions = {
            "train_loss": WeakCombinedLoss(config["train_loss_parameters"]),
            "val_loss": WeakCombinedLoss(config["val_loss_parameters"]),
        }
        self.metrics_functions = {
            "train_metrics": MetricsCalculator(config["train_metrics_parameters"]),
            "val_metrics": MetricsCalculator(config["val_metrics_parameters"]),
        }
        self.transforms = DataTransforms(config["transform_parameters"])

    def load_best_model(self):
        checkpoint_path = (
            f"{self.config['runs_path']}/{self.experiment.exp_name}/models/best.pth"
        )
        self.net.load_state_dict(
            torch.load(checkpoint_path, map_location=self.config["device"])
        )
        self.net.to(self.config["device"])

    def reset_epoch_counters(self):
        self.vars["actual_epoch_loss"] = 0.0
        self.vars["actual_num_of_step"] = 0.0

    def train_step(self, images, true_masks):
        self.vars["masks_pred"] = self.net(images)
        loss = self.loss_functions["train_loss"](self.vars["masks_pred"], true_masks)
        loss.backward()
        self.optimizer.step()
        return loss

    def val_step(self, images, true_masks):
        with torch.no_grad():
            self.vars["masks_pred"] = self.net(images)
            loss = self.loss_functions["val_loss"](self.vars["masks_pred"], true_masks)
        return loss

    def start_epoch(self, data_loader, phase="train"):
        self.vars["phase"] = phase
        self.net.train() if phase == "train" else self.net.eval()
        self.reset_epoch_counters()

        with tqdm(
            total=len(data_loader),
            desc=f'Epoch {self.vars["epoch"] + 1}/{self.config["epochs"]}',
            unit="img",
        ) as pbar:
            for batch in data_loader:
                images, true_masks = batch["images"], batch["masks"]
                images, true_masks = (
                    images.to(self.config["device"]),
                    true_masks.to(self.config["device"]),
                )
                loss = (
                    self.train_step(images, true_masks)
                    if phase == "train"
                    else self.val_step(images, true_masks)
                )

                self.vars["actual_loss"] = loss.item()
                self.vars["actual_epoch_loss"] += loss.item()
                self.vars["actual_num_of_step"] += 1
                pbar.set_postfix(**{"loss (batch)": self.vars["actual_loss"]})
                pbar.update(1)

    def train(self):
        for epoch in range(self.config["epochs"]):
            self.vars["epoch"] = epoch
            self.start_epoch(self.train_dataloader, phase="train")
            self.start_epoch(self.val_dataloader, phase="val")
            self.check_early_stopping()

    def check_early_stopping(self):
        if self.vars["actual_epoch_loss"] < self.vars["best_epoch_loss"]:
            self.vars["best_epoch_loss"] = self.vars["actual_epoch_loss"]
            self.save_model("best")

    def save_model(self, name):
        model_path = (
            f"{self.config['runs_path']}/{self.experiment.exp_name}/models/{name}.pth"
        )
        torch.save(self.net.state_dict(), model_path)


config = {
    "exp_name": "example_experiment",
    "runs_path": "./experiments",
    "epochs": 10,
    "early_stop": 3,
    "train_loss_parameters": {"param1": 1, "param2": 0.5},
    "val_loss_parameters": {"param1": 1, "param2": 0.5},
    "train_metrics_parameters": {"metric1": "accuracy"},
    "val_metrics_parameters": {"metric1": "accuracy"},
    "transform_parameters": {"resize": 256, "normalize": True},
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

trainer = Trainer(
    net=model, train_dataloader=train_loader, val_dataloader=val_loader, config=config
)
trainer.train()

######################################################

##################################################
##################################################
##################################################
##################################################
##################################################
##################################################
##################################################


class ExperimentSetup:
    def __init__(self, config):
        self.config = config
        self.exp_name = self.create_experiment_dir()

    def create_experiment_dir(self):
        # пока так оставлю
        # timestamp = time.strftime("%Y-%m-%d__%H_%M_%S")
        # exp_name = f"{timestamp}_{self.config['exp_name']}"
        exp_name = f"{self.config['exp_name']}"
        os.makedirs(f"{self.config['runs_path']}/{exp_name}", exist_ok=True)
        return exp_name


class BaseTrainer(ABC):
    def __init__(self, dataloaders, config):
        self.train_loader = dataloaders["train"]
        self.val_loader = dataloaders["val"]
        self.config = config
        self.device = config["device"]
        self.model = config["model"].to(self.device)
        self.experiment = ExperimentSetup(config)

    @abstractmethod
    def train_one_epoch(self, epoch):
        pass

    @abstractmethod
    def validate(self, epoch):
        pass

    # сделать сохранение по лучшей эпохе и тд
    def save_model(self, name):  # save_checkpoint можно назвать
        model_path = (
            f"{self.config['runs_path']}/{self.experiment.exp_name}/models/{name}.pth"
        )
        torch.save(self.model.state_dict(), model_path)

    def train(self):
        for epoch in range(self.config["epochs"]):
            self.train_one_epoch(epoch)
            self.validate(epoch)
            # self.save_model(epoch)


class SegmentationTrainer(BaseTrainer):
    def __init__(self, dataloaders, config):
        super().__init__(dataloaders, config)
        # self.optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        # надо тут заменить потом везде config["model"]
        self.optimizer = config["optimizer"](
            config["model"].parameters(), lr=config["lr"]
        )

        # self.device = config["device"]
        print("self.device", self.device)

        # это тоже менять надо, лосс должен в конфиге задаваться
        self.criterion = WeakCombinedLoss(*config["train_loss_parameters"])
        num_classes = 3

        self.writer = SummaryWriter(
            log_dir=f"runs_kidneys/{self.experiment.exp_name}_logs"
        )
        self.metrics_calculator_train = DetectionMetrics(
            mode="ML", num_classes=num_classes
        )
        self.metrics_calculator_val = DetectionMetrics(
            mode="ML", num_classes=num_classes
        )

        self.class_names_dict = {
            class_info["id"]: class_info["name"]
            for class_info in kidneys_pat_out_classes
        }
        print("class_names_dict", self.class_names_dict)

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        with tqdm(
            total=len(self.train_loader), desc=f"Training Epoch {epoch+1}", unit="batch"
        ) as pbar:
            for batch in self.train_loader:
                images, masks = (
                    batch["images"].to(self.device),
                    batch["masks"][:, 1:, :, :].to(self.device),
                )

                self.optimizer.zero_grad()
                predictions = self.model(images)
                # тут надо подумать как лучше сделать
                predictions = torch.sigmoid(predictions)
                loss = self.criterion.forward(predictions, masks)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                self.metrics_calculator_train.update_counter(
                    masks,
                    predictions,
                )

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        train_loss_avg = total_loss / len(self.train_loader)
        print(f"Training Loss Epoch {epoch + 1}: {train_loss_avg}")

        self.writer.add_scalar("Loss/train", train_loss_avg, epoch)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            train_metrics = self.metrics_calculator_train.calc_metrics()

        for key, value in train_metrics.items():
            if isinstance(value, torch.Tensor):
                if len(value.size()) > 0:  # Проверяем, что тензор не пустой
                    # средняя метрика
                    self.writer.add_scalar(
                        f"Train/Mean/{key}", value.mean().item(), epoch
                    )
                    for i, val in enumerate(value):
                        class_name = self.class_names_dict[i + 1]
                        # writer.add_scalar(f"Train/{key}/Class_{i}", val.item(), epoch)
                        self.writer.add_scalar(
                            f"Train/{key}/{class_name}", val.item(), epoch
                        )
                else:
                    self.writer.add_scalar(f"Train/{key}", value.item(), epoch)

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            with tqdm(
                total=len(self.val_loader),
                desc=f"Validation Epoch {epoch+1}",
                unit="batch",
            ) as pbar:
                for batch in self.val_loader:
                    images, masks = (
                        batch["images"].to(self.device),
                        batch["masks"][:, 1:, :, :].to(self.device),
                    )

                    predictions = self.model(images)
                    # тут надо подумать как лучше сделать
                    predictions = torch.sigmoid(predictions)
                    loss = self.criterion.forward(predictions, masks)

                    self.metrics_calculator_val.update_counter(
                        masks,
                        predictions,
                    )

                    total_loss += loss.item()
                    pbar.update(1)

        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss Epoch {epoch + 1}: {avg_loss}")

        # надо save_model делать
        self.save_model(...)
        # а это убрать
        if avg_loss < self.config.get("best_loss", float("inf")):
            self.config["best_loss"] = avg_loss
            self.save_model("best")

        self.writer.add_scalar("Loss/validation", avg_loss, epoch)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            val_metrics = self.metrics_calculator_val.calc_metrics()

        for key, value in val_metrics.items():
            if isinstance(value, torch.Tensor):
                if len(value.size()) > 0:
                    # добавил среднюю метрику по классам
                    self.writer.add_scalar(
                        f"Val/Mean/{key}", value.mean().item(), epoch
                    )
                    for i, val in enumerate(value):
                        class_name = self.class_names_dict[i + 1]
                        # writer.add_scalar(f"Val/{key}/Class_{i}", val.item(), epoch)
                        self.writer.add_scalar(
                            f"Val/{key}/{class_name}", val.item(), epoch
                        )
                else:
                    self.writer.add_scalar(f"Val/{key}", value.item(), epoch)


class Florence2DetectionTrainer(BaseTrainer):
    def __init__(self, model, dataloaders, config, processor):
        super().__init__(model, dataloaders, config)
        self.processor = processor
        self.optimizer = AdamW(model.parameters(), lr=config["lr"])
        self.criterion = FocalLoss()

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        with tqdm(
            total=len(self.train_loader),
            desc=f"Training Epoch {epoch + 1}",
            unit="batch",
        ) as pbar:
            for inputs, answers in self.train_loader:
                labels = self.processor.tokenizer(
                    answers, return_tensors="pt"
                ).input_ids.to(self.device)
                outputs = self.model(**inputs.to(self.device), labels=labels)
                logits = outputs.logits.view(-1, outputs.logits.size(-1))
                labels = labels.view(-1)
                loss = self.criterion.forward(logits, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
        print(f"Training Loss Epoch {epoch + 1}: {total_loss / len(self.train_loader)}")

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            with tqdm(
                total=len(self.val_loader),
                desc=f"Validation Epoch {epoch + 1}",
                unit="batch",
            ) as pbar:
                for inputs, answers in self.val_loader:
                    labels = self.processor.tokenizer(
                        answers, return_tensors="pt"
                    ).input_ids.to(self.device)
                    outputs = self.model(**inputs.to(self.device), labels=labels)
                    logits = outputs.logits.view(-1, outputs.logits.size(-1))
                    labels = labels.view(-1)
                    loss = self.criterion.forward(logits, labels)
                    total_loss += loss.item()
                    pbar.update(1)
        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss Epoch {epoch + 1}: {avg_loss}")
        if avg_loss < self.config.get("best_loss", float("inf")):
            self.config["best_loss"] = avg_loss
            self.save_model("best")


# отсюда можно что то в новый взять потому что мне не всегда надо делать решейпы, зависит от лосса


class old_Florence2DetectionTrainer(BaseTrainer):
    def __init__(self, model, dataloaders, config, processor):
        super().__init__(model, dataloaders, config)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
        self.processor = processor

    def train_one_epoch(self, epoch):
        self.model.train()
        for inputs, answers in self.train_loader:
            labels = self.processor.tokenizer(answers, return_tensors="pt").input_ids
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

    def validate(self, epoch):
        # Реализация валидации для Florence2 детекции
        pass

    def save_checkpoint(self, epoch):
        # Сохранение модели
        pass


class TrainerFactory:
    @staticmethod
    def create_trainer(task, dataloaders, config, processor=None):
        if task == "segmentation":
            return SegmentationTrainer(dataloaders, config)
        elif task == "detection" and config.get("model_name") == "Florence2":
            return Florence2DetectionTrainer(
                config["model"], dataloaders, config, processor
            )
        else:
            raise ValueError(
                f"Trainer for task '{task}' and model '{config.get('model_name')}' not implemented"
            )


if __name__ == "__main__":
    batch_size = 6
    num_classes = 3

    params = {
        "json_file_path": "/home/imran-nasyrov/json_pochki",
        "delete_list": [],
        "base_classes": kidneys_base_classes,
        # kidneys_out_classes или kidneys_pat_out_classes
        "out_classes": kidneys_pat_out_classes,
        "dataloader": True,
        "resize": (512, 512),
        "recalculate": False,
        "delete_null": False,
    }

    coco_dataloader = SINUSITE_COCODataLoader(params)

    (
        train_loader,
        val_loader,
        total_train,
        pixel_total_train,
        list_of_name_out_classes,
    ) = coco_dataloader.make_dataloaders(batch_size=batch_size, train_val_ratio=0.8)

    model = smp.Linknet(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=1,  # +num_classes для диффузии
        classes=num_classes,
    )

    print("total_train", total_train)
    print("len total_train", len(total_train))
    print("list_of_name_out_classes", list_of_name_out_classes)
    print("pixel_TotalTrain", pixel_total_train)
    print("len val_loader", len(val_loader))
    print("len train_loader", len(train_loader))

    device = torch.device("cuda:0")
    print(device)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    config = {
        "task": "segmentation",
        "model": model,
        "epochs": 120,
        "train_loss_parameters": (None, None),
        "optimizer": torch.optim.Adam,
        "lr": 3e-4,
        "exp_name": "kidneys_new_code_1.10",
        "runs_path": "home/imran-nasyrov/runs_kidneys",
        "device": device,
    }

    trainer = TrainerFactory.create_trainer(
        config["task"],
        {"train": train_loader, "val": val_loader},
        config,
        # processor надо убрать отсюда потом
        # processor=processor,
    )
    trainer.train()
