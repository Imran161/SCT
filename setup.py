import torch
import segmentation_models_pytorch as smp

from src.metrics.metrics import DetectionMetrics
from src.trainers.abstract_trainer import AbstractTrainer
from src.trainers.trainer_functions import (
    standart_batch_function,
    standart_logging_manager,
    standart_weight_saving_manager,
)
from src.model_factory import ModelFactory  # Убедитесь, что путь правильный
from src.datamanager.coco_dataloaders import SINUSITE_COCODataLoader
from src.datamanager.coco_classes import (
    kidneys_base_classes,
    kidneys_pat_out_classes,
)

if __name__ == "__main__":
    batch_size = 6

    params = {
        "json_file_path": "/home/imran-nasyrov/json_pochki",
        "delete_list": [],
        "base_classes": kidneys_base_classes,
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

    dataloaders = {"train": train_loader, "val": val_loader}

    print("total_train", total_train)
    print("len total_train", len(total_train))
    print("list_of_name_out_classes", list_of_name_out_classes)
    print("pixel_TotalTrain", pixel_total_train)
    print("len val_loader", len(val_loader))
    print("len train_loader", len(train_loader))

    device = torch.device("cuda:0")
    print(device)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    model = smp.Linknet(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=1,  # +num_classes для диффузии
        classes=3,
    )

    config = {
        "task": "segmentation",
        "model": "Linknet",
        "encoder_name": "efficientnet-b7",
        "encoder_weights": "imagenet",
        "in_channels": 1,
        "num_classes": 3,
        "epochs": 120,
        "phases": ["train", "val"],
        "activation": "sigmoid",
        "save_path": "best_model.pth",
        "train_loss_parameters": (None, None),
        "optimizer": torch.optim.Adam(model.parameters(), lr=3e-4),
        "metrics": DetectionMetrics(),
    }

    model = ModelFactory.create_model(config)
    trainer = AbstractTrainer(dataloaders, config)

    trainer.start_training()
