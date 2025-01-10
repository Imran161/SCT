import torch
import segmentation_models_pytorch as smp
import numpy as np

from src.metrics.metrics import DetectionMetrics
from src.trainers.abstract_trainer import AbstractTrainer
from src.setup_tools.experiment_setup import ExperimentSetup
from src.trainers.trainer_functions import (
    standart_logging_manager,
    standart_weight_saving_manager,
)
from src.trainers.model_factory import ModelFactory
from src.datamanager.coco_dataloaders import SINUSITE_COCODataLoader, KIDNEYS_COCODataLoader
from src.datamanager.coco_classes import (
    kidneys_base_classes,
    kidneys_pat_out_classes,
    kidneys_segment_out_classes
)


class Weight_opt_class:
    def __init__(self, loss, classes, b=None):
        self.b = b
        self.loss = loss
        self.loss_class = loss
        self.classes = classes

    # оптимизация новых метрик
    def opt_pixel_weight(self, metrics, pixel_all_class_weights=None):
        recall = metrics["advanced_recall"]
        precession = metrics["advanced_precision"]  # раньше precession было
        F1Score = metrics["advanced_F1"]

        b = self.b

        if b is None:
            b = 1

        for image_class, cl_name in enumerate(self.classes):
            neg_coef = 1
            pos_coef = 1

            if recall[image_class].item() != 0 and precession[image_class].item() != 0:
                print("recall и precision != 0")
                print("recall[image_class].item()", recall[image_class].item())
                print("precession[image_class].item()", precession[image_class].item())

                neg_coef = (
                    (1 / b) * recall[image_class].item() / F1Score[image_class].item()
                )
                pos_coef = (
                    (b) * precession[image_class].item() / F1Score[image_class].item()
                )
                print("neg_coef", neg_coef)
                print("pos_coef", pos_coef)

                xsd = recall[image_class].item() / precession[image_class].item()
                print("xsd", xsd)
                if xsd > 0.9 and xsd < 1.1:
                    neg_coef = 1
                    pos_coef = 1
                class_coef = pos_coef
                print("вот после изменений")
                print("neg_coef", neg_coef)
                print("pos_coef", pos_coef)
                print("class_coef", class_coef)

            else:
                print("recall или precision == 0")
                print("recall[image_class].item()", recall[image_class].item())
                print("precession[image_class].item()", precession[image_class].item())

                pos_coef = 2.0
                class_coef = 2.0
                neg_coef = 0.5
                print("neg_coef", neg_coef)
                print("pos_coef", pos_coef)
                print("class_coef", class_coef)

            if pixel_all_class_weights is not None:
                pixel_all_class_weights[0][image_class] *= pos_coef
                pixel_all_class_weights[1][image_class] *= neg_coef
                pixel_all_class_weights[2][image_class] *= class_coef

        return pixel_all_class_weights


if __name__ == "__main__":
    batch_size = 24
    num_classes = 6

    params = {
        "json_file_path": "/home/imran/data/mini_dataset", # "/home/imran/data/29_11_split_kidneys",
        "delete_list": [],
        "base_classes": kidneys_base_classes,
        "out_classes": kidneys_segment_out_classes,
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

    device = torch.device("cuda:0")
    print(device)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    use_class_weight = True
    use_pixel_weight = True
    use_pixel_opt = True
    power = "26_12_test_code_weak"
    
    exp_setup = ExperimentSetup(
        train_loader, total_train, pixel_total_train, batch_size, num_classes
    )
     
    (
        all_class_weights,
        pixel_all_class_weights,
        experiment_name,
        criterion,
    ) = exp_setup.setup_experiment(
        use_class_weight, use_pixel_weight, use_pixel_opt, power
    )
    
    if all_class_weights is not None:
        all_weights_no_fon = [x[1:] for x in all_class_weights]
    else:
        all_weights_no_fon = None
        
    if pixel_all_class_weights is not None:
        alpha_no_fon = np.array([arr[1:] for arr in pixel_all_class_weights])
        alpha_no_fon = torch.tensor(alpha_no_fon).to(device)
    else:
        alpha_no_fon = None

    class_names_dict = {
        class_info["id"]: class_info["name"]
        for class_info in kidneys_segment_out_classes
    }
    print("class_names_dict", class_names_dict)

    classes = list(class_names_dict.keys())
    
    if use_pixel_opt:
        weight_opt = Weight_opt_class(criterion, classes, None)
                    
    config = {
        "device": device,
        "task": "segmentation",
        "model": "Linknet",
        "encoder_name": "efficientnet-b7",
        "encoder_weights": "imagenet",
        "in_channels": 1,
        "num_classes": num_classes,
        "epochs": 120,
        "phases": ["train", "val"],
        "activation": torch.sigmoid,
        # "save_path": "SCT/best_models/best_model.pth",
        # loss: ..., 
        "train_loss_parameters": (all_weights_no_fon, alpha_no_fon),
        "alpha_no_fon": alpha_no_fon,
        "weight_opt": weight_opt,
        # "optimizer": torch.optim.Adam,
        "metrics": DetectionMetrics,
        "model_save_dir": "test_new_code/runs_kidneys",
        "experiment_name": experiment_name,
    }

    trainer = AbstractTrainer(dataloaders, config)
    trainer.start_training()
