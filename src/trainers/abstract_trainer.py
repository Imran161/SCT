import torch
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from src.trainers.trainer_functions import (
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

from ..datamanager.coco_classes import (
    kidneys_base_classes,
    kidneys_pat_out_classes,
    kidneys_segment_out_classes
)


class_names_dict = {
    class_info["id"]: class_info["name"]
    for class_info in kidneys_segment_out_classes
}
print("class_names_dict", class_names_dict)

classes = list(class_names_dict.keys())

class AbstractTrainer:
    def __init__(self, dataloaders, config, rank=0, world_size=1):
        self.dataloaders = dataloaders
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")

        # Initialize the process group for distributed training
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

        self.model = ModelFactory.create_model(config).to(self.device)
        self.model = DistributedDataParallel(self.model, device_ids=[rank])

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        self.variables = {
            "device": self.config.get("device", None),
            "num_classes": self.config.get("num_classes", 3),
            "alpha_no_fon": self.config.get("alpha_no_fon", None),
            "weight_opt": self.config.get("weight_opt", None),
            "current_epoch": 0,
            "current_phase": None,
            "current_loss": 0,
            "epoch_loss": 0,
            "weight": None,
            "outputs": None,
            "phase_losses": {"train": 0.0, "val": 0.0},
        }

        self.functions = {
            # "model": ModelFactory.create_model(config),
            "configurate_model": standart_model_configurate,
            "hyperparam_manager": None,
            "logging_manager": standart_logging_manager,
            "log_metrics": log_metrics,
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
        
        # self.functions["optimizer"] = self.config.get(
        #     "optimizer", torch.optim.Adam(self.functions["model"].parameters(), lr=3e-4)
        #
        # )
            

    def start_training(self):
        target_epoch = self.config["epochs"]
        self.variables["current_epoch"] = 0
        # self.functions["model"] = self.functions["model"].to(self.variables["device"])

        experiment_name = self.config["experiment_name"]
        writer = self.functions["logging_manager"](self.config, experiment_name)

        while self.variables["current_epoch"] < target_epoch:
            for phase in self.config["phases"]:

                if phase == 'train' and self.dataloaders[phase].sampler is not None: ######
                    self.dataloaders[phase].sampler.set_epoch(self.variables["current_epoch"]) #####

                self.variables["current_phase"] = phase
                dataloader = self.dataloaders[self.variables["current_phase"]]

                self.functions["configurate_model"](
                    self.functions["model"], self.variables
                )
                self.variables["epoch_loss"] = 0
                self.variables["phase_losses"][phase] = 0.0

                with tqdm(total=len(dataloader), desc=f"Epoch {self.variables['current_epoch']} Phase {phase}", unit="batch") as pbar:
                    # for batch in dataloader: так было
                    #     self.process_batch(batch)
                    #     pbar.set_postfix(**{'loss (batch)': self.variables["epoch_loss"].item()})
                    #     pbar.update(1)
                    for batch in dataloader:
                        self.process_batch(batch)
                        pbar.set_postfix(loss=self.variables["current_loss"].item() / (pbar.n + 1))
                        pbar.update(1)

                # print(f"{phase.capitalize()} Average Loss: {self.variables['phase_losses'][phase]}")


                self.reduce_epoch_loss()

                if self.rank == 0:
                    print(f"Epoch {self.variables['current_epoch']} Phase {phase} - Global Loss: {self.variables['global_epoch_loss']}")


                metrics = self.compute_epoch_metrics(phase)
                self.functions["log_metrics"](writer, phase, metrics, self.variables["current_epoch"], self.variables["phase_losses"][phase])
                self.functions["weight_saving_manager"](self.variables, self.functions["model"], self.config)
                
                print("metrics", metrics)
                
                if self.variables["weight_opt"]:
                    self.variables["alpha_no_fon"] = self.variables["weight_opt"].opt_pixel_weight(self.variables["train_metrics"], self.variables["alpha_no_fon"])

                if self.variables["alpha_no_fon"] is not None:
                    # print("class", class_names_dict[1])
                    # print("alpha_no_fon pixel_pos_weights", alpha_no_fon[0])

                    print(
                        f"\nclass: {class_names_dict[1]}, pixel_pos_weights {self.variables['alpha_no_fon'][0][0]}"
                    )
                    print(
                        f"class: {class_names_dict[2]}, pixel_pos_weights {self.variables['alpha_no_fon'][0][1]}"
                    )
                    print(
                        f"class: {class_names_dict[3]}, pixel_pos_weights {self.variables['alpha_no_fon'][0][2]}\n"
                    )

                    print(
                        f"class: {class_names_dict[1]}, pixel_neg_weights {self.variables['alpha_no_fon'][1][0]}"
                    )
                    print(
                        f"class: {class_names_dict[2]}, pixel_neg_weights {self.variables['alpha_no_fon'][1][1]}"
                    )
                    print(
                        f"class: {class_names_dict[3]}, pixel_neg_weights {self.variables['alpha_no_fon'][1][2]}\n"
                    )

                    print(
                        f"class: {class_names_dict[1]}, pixel_class_weights {self.variables['alpha_no_fon'][2][0]}"
                    )
                    print(
                        f"class: {class_names_dict[2]}, pixel_class_weights {self.variables['alpha_no_fon'][2][1]}"
                    )
                    print(
                        f"class: {class_names_dict[3]}, pixel_class_weights {self.variables['alpha_no_fon'][2][2]}\n"
                    )


            self.variables["current_epoch"] += 1



    @no_grad_for_validation
    def process_batch(self, batch):
        inputs = batch["images"].to(self.variables["device"])
        targets = batch["masks"][:, 1:, :, :].to(self.variables["device"])

        outputs = self.predict(inputs)
        outputs = self.functions["activation"](outputs).clone()

        self.update_loss(targets, outputs)
        self.variables["phase_losses"][self.variables["current_phase"]] += self.variables["current_loss"] / len(self.dataloaders[self.variables["current_phase"]])

        self.functions["metrics_calculator"].update_counter(targets, outputs)
        self.optimize_step()



    def optimize_step(self):
        if self.variables["current_phase"] == "train":
            self.functions["optimizer"].zero_grad()
            self.variables["current_loss"].backward()
            self.functions["optimizer"].step()
           
           
    def update_loss(self, inputs, outputs):
        # if self.functions["loss_function"]:
        #     self.variables["current_loss"] = self.functions["loss_function"].forward(
        #         inputs, outputs
        #     )
        #     self.variables["epoch_loss"] += self.variables["current_loss"] / len(self.dataloaders[self.variables["current_phase"]]) # если есть это? Добавил / len(....)

        if self.functions["loss_function"]:
            self.variables["current_loss"] = self.functions["loss_function"].forward(inputs, outputs)


    def compute_epoch_metrics(self, phase):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            metrics = self.functions["metrics_calculator"].calc_metrics()

        print(f"{phase.capitalize()} metrics: {metrics}")
        self.variables[f"{phase}_metrics"] = metrics
        return metrics
    
    
    def predict(self, inputs):
        return self.functions["model"](inputs)


    def reduce_epoch_loss(self):
        """Вычисляет средний глобальный лосс по всем процессам."""
        loss_tensor = torch.tensor(self.variables["epoch_loss"], device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        self.variables["global_epoch_loss"] = loss_tensor.item() / self.world_size
