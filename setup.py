import torch
import segmentation_models_pytorch as smp

from src.metrics.metrics import DetectionMetrics
from src.trainers.abstract_trainer import AbstractTrainer
from src.trainers.trainer_functions import (
    standart_batch_function,
    standart_logging_manager,
    standart_weight_saving_manager,
)
from src.trainers.model_factory import ModelFactory
from src.datamanager.coco_dataloaders import SINUSITE_COCODataLoader, KIDNEYS_COCODataLoader
from src.datamanager.coco_classes import (
    kidneys_base_classes,
    kidneys_pat_out_classes,
)

if __name__ == "__main__":
    batch_size = 6

    params = {
        "json_file_path": "/home/imran-nasyrov/export_pochky_code/29_11_split_kidneys",
        "delete_list": [],
        "base_classes": kidneys_base_classes,
        "out_classes": kidneys_pat_out_classes,
        "dataloader": True,
        "resize": (512, 512),
        "recalculate": False,
        "delete_null": False,
    }

    coco_dataloader = KIDNEYS_COCODataLoader(params)

    (
        train_loader,
        val_loader,
        total_train,
        pixel_total_train,
        list_of_name_out_classes,
    ) = coco_dataloader.make_dataloaders(batch_size=batch_size)

    dataloaders = {"train": train_loader, "val": val_loader}

    print("total_train", total_train)
    print("len total_train", len(total_train))
    print("list_of_name_out_classes", list_of_name_out_classes)
    print("pixel_TotalTrain", pixel_total_train)
    print("len val_loader", len(val_loader))
    print("len train_loader", len(train_loader))

    device = torch.device("cuda:2")
    print(device)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    config = {
        "device": device,
        "task": "segmentation",
        "model": "Linknet",
        "encoder_name": "efficientnet-b7",
        "encoder_weights": "imagenet",
        "in_channels": 1,
        "num_classes": 3,
        "epochs": 120,
        "phases": ["train", "val"],
        "activation": torch.sigmoid,
        "save_path": "SCT/best_models/best_model.pth",
        # loss: ..., 
        "train_loss_parameters": (None, None),
        # "optimizer": torch.optim.Adam,
        "metrics": DetectionMetrics,
        "model_save_dir": "test_new_code/runs_kidneys",
        "experiment_name": "test",
    }

    trainer = AbstractTrainer(dataloaders, config)
    trainer.start_training()
