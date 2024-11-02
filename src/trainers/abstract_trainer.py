import torch
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils.functions import (
    apply_activation,
    standart_batch_function,
    standart_logging_manager,
    standart_model_configurate,
    standart_weight_saving_manager,
)

from ..datamanager.coco_classes import (
    kidneys_base_classes,
    kidneys_pat_out_classes,
)
from ..datamanager.coco_dataloaders import SINUSITE_COCODataLoader
from ..losses.losses_cls import (
    FocalLoss,
    WeakCombinedLoss,
)  # Подключи кастомные функции лосса
from ..metrics.metrics import DetectionMetrics
from .model_factory import ModelFactory


class AbstractTrainer:
    def __init__(self, dataloaders, config):
        self.dataloaders = dataloaders
        self.config = config

        self.variables = {
            "current_epoch": 0,
            "current_phase": None,
            "current_loss": 0,
            "epoch_loss": 0,
            "weight": None,
            "outputs": None,
        }

        self.functions = {
            "model": ModelFactory.create_model(config),
            "get_batch": standart_batch_function,
            "get_batch": standart_batch_function,
            "configurate_model": standart_model_configurate,
            "hyperparam_manager": None,
            "logging_manager": standart_logging_manager,
            "weight_saving_manager": standart_weight_saving_manager,
            "optimizer": self.config.get("optimizer", torch.optim.Adam),
            "loss_function": self.config.get(
                "loss", WeakCombinedLoss(*config["train_loss_parameters"])
            ),
            "metrics": self.config.get("metrics", DetectionMetrics()),
        }

    def start_training(self):
        target_epoch = self.config["epochs"]
        self.variables["current_epoch"] = 0

        while self.variables["current_epoch"] < target_epoch:
            for phase in self.config["phases"]:
                self.variables["current_phase"] = phase
                dataloader = self.dataloaders[self.variables["current_phase"]]

                # Конфигурация модели в зависимости от фазы
                self.functions["configurate_model"](
                    self.functions["model"], self.variables
                )
                self.variables["epoch_loss"] = 0

                for batch in dataloader:
                    self.process_batch(batch)

                # Проверка на существование функций перед вызовом
                if self.functions.get("hyperparam_manager"):
                    self.functions["hyperparam_manager"](self.variables, self.functions)
                if self.functions.get("logging_manager"):
                    self.functions["logging_manager"](self.variables, self.functions)
                if self.functions.get("weight_saving_manager"):
                    self.functions["weight_saving_manager"](
                        self.variables, self.functions["model"], self.config
                    )

            self.variables["current_epoch"] += 1

    def process_batch(self, batch):
        data = self.functions["get_batch"](batch)
        inputs, targets = (
            data["images"],
            data["masks"],
        )

        outputs = self.predict(inputs)
        outputs = apply_activation(outputs, self.config)

        # Обновление потерь и метрик
        self.update_loss(inputs, outputs)
        self.update_metrics(inputs, outputs)

        if self.variables["current_phase"] == "train":
            self.functions["optimizer"].zero_grad()
            self.variables["current_loss"].backward()
            self.functions["optimizer"].step()

    def update_loss(self, inputs, outputs):
        # Проверка, определена ли функция потерь
        if self.functions["loss_function"]:
            self.variables["current_loss"] = self.functions["loss_function"](
                inputs, outputs, self.variables["weight"]
            )
            self.variables["epoch_loss"] += self.variables["current_loss"]

    def update_metrics(self, inputs, outputs):
        # Проверка, определена ли метрика
        if self.functions["metrick"]:
            self.functions["metrick"].update_counter(inputs, outputs)

    def predict(self, inputs):
        """
        Определяет способ получения предсказаний модели.
        Может быть переопределен в подклассах или передан как функция.
        """
        return self.functions["model"](inputs)  # По умолчанию прямой вызов
