import torch
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.functions import (
    standart_logging_manager,
    log_metrics,
    standart_model_configurate,
    standart_weight_saving_manager,
    no_grad_for_validation
)

from ..datamanager.coco_classes import (
    kidneys_base_classes,
    kidneys_pat_out_classes,
)
from ..datamanager.coco_dataloaders import SINUSITE_COCODataLoader, KIDNEYS_COCODataLoader
from ..losses.losses_cls import (
    FocalLoss,
    WeakCombinedLoss,
)
from ..metrics.metrics import DetectionMetrics
from .model_factory import ModelFactory


class AbstractTrainer:
    def __init__(self, dataloaders, config):
        self.dataloaders = dataloaders
        self.config = config

        self.variables = {
            "device": self.config.get("device", None),
            "num_classes": self.config.get("num_classes", 3),
            "current_epoch": 0,
            "current_phase": None,
            "current_loss": 0,
            "epoch_loss": 0,
            "weight": None,
            "outputs": None,
            "phase_losses": {"train": 0.0, "val": 0.0},
        }

        self.functions = {
            "model": ModelFactory.create_model(config),
            "configurate_model": standart_model_configurate,
            "hyperparam_manager": None,
            "logging_manager": standart_logging_manager,
            "weight_saving_manager": standart_weight_saving_manager,
            "activation" : self.config.get("activation", torch.sigmoid),
            "loss_function": self.config.get(
                "loss", WeakCombinedLoss(*config["train_loss_parameters"])
            ),
            "metrics": self.config.get("metrics", DetectionMetrics),
            "metrics_calculator": DetectionMetrics(
                mode="ML", num_classes=self.config["num_classes"]
            ),
        }
        
        self.functions["optimizer"] = self.config.get(
            "optimizer", torch.optim.Adam(self.functions["model"].parameters(), lr=3e-4)
        )
            

    def start_training(self):
        target_epoch = self.config["epochs"]
        self.variables["current_epoch"] = 0
        self.functions["model"] = self.functions["model"].to(self.variables["device"])

        experiment_name = self.config["experiment_name"]
        writer = self.functions["logging_manager"](self.config, experiment_name)

        while self.variables["current_epoch"] < target_epoch:
            for phase in self.config["phases"]:
                self.variables["current_phase"] = phase
                dataloader = self.dataloaders[self.variables["current_phase"]]

                self.functions["configurate_model"](
                    self.functions["model"], self.variables
                )
                self.variables["epoch_loss"] = 0
                self.variables["phase_losses"][phase] = 0.0

                for batch in tqdm(dataloader, desc=f"Epoch {self.variables['current_epoch']} Phase {phase}"):
                    self.process_batch(batch)

                average_phase_loss = self.variables["phase_losses"][phase] / len(dataloader)
                print(f"{phase.capitalize()} Average Loss: {average_phase_loss}")

                metrics = self.compute_epoch_metrics(phase)
                self.functions["log_metrics"](writer, phase, metrics, self.variables["current_epoch"])
                self.functions["weight_saving_manager"](self.variables, self.functions["model"], self.config)

            self.variables["current_epoch"] += 1



    @no_grad_for_validation
    def process_batch(self, batch):
        inputs = batch["images"].to(self.variables["device"])
        targets = batch["masks"][:, 1:, :, :].to(self.variables["device"])

        outputs = self.predict(inputs)
        outputs = self.functions["activation"](outputs).clone()

        self.update_loss(targets, outputs)
        self.variables["epoch_loss"] += self.variables["current_loss"].item()
        current_phase = self.variables["current_phase"]
        self.variables["phase_losses"][current_phase] += self.variables["current_loss"].item()

        self.functions["metrics_calculator"].update_counter(targets, outputs)
        self.optimize_step()


    def optimize_step(self):
        if self.variables["current_phase"] == "train":
            self.functions["optimizer"].zero_grad()
            self.variables["current_loss"].backward()
            self.functions["optimizer"].step()
           
           
    def update_loss(self, inputs, outputs):
        if self.functions["loss_function"]:
            self.variables["current_loss"] = self.functions["loss_function"].forward(
                inputs, outputs
            )
            self.variables["epoch_loss"] += self.variables["current_loss"]


    def compute_epoch_metrics(self, phase):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            metrics = self.functions["metrics_calculator"].calc_metrics()

        print(f"{phase.capitalize()} metrics: {metrics}")
        self.variables[f"{phase}_metrics"] = metrics
        return metrics
    
    
    def predict(self, inputs):
        return self.functions["model"](inputs)
