import torch
import os

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

def standart_weight_saving_manager(variables, model, config):
    save_dir = os.path.join(config["model_save_dir"], "models", f"{config['experiment_name']}")
    os.makedirs(save_dir, exist_ok=True)

    current_model_path = os.path.join(save_dir, f"model_epoch_{variables['current_epoch']}.pth")
    torch.save(model.state_dict(), current_model_path)
    print(f"Сохранена текущая модель в {current_model_path} на эпохе {variables['current_epoch']}")

    if "best_val_loss" not in variables or variables["phase_losses"]["val"] < variables["best_val_loss"]:
        variables["best_val_loss"] = variables["phase_losses"]["val"]
        best_model_path = os.path.join(save_dir, "best_model.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"Сохранена лучшая модель с ошибкой на валидации: {variables['best_val_loss']}")


def standart_model_configurate(model, variables):
    if variables["current_phase"] == "train":
        model.train()
    elif variables["current_phase"] == "val":
        model.eval()
    else:
        raise ValueError(f'Фаза "{variables["current_phase"]}" не поддерживается')
    
    
def no_grad_for_validation(func):
    def wrapper(self, *args, **kwargs):
        if self.variables["current_phase"] == "val":
            with torch.no_grad():
                return func(self, *args, **kwargs)
        return func(self, *args, **kwargs)
    return wrapper


def standart_logging_manager(config, experiment_name):
    from torch.utils.tensorboard import SummaryWriter
    log_dir = os.path.join(config["model_save_dir"], f"{experiment_name}_logs")
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)


def log_metrics(writer, phase, metrics, epoch, losses):
    writer.add_scalar(f"{phase}/Loss", losses, epoch)

    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if len(value.size()) > 0:
                writer.add_scalar(f"{phase}/Mean/{key}", value.mean().item(), epoch)
                for i, val in enumerate(value):
                    class_name = class_names_dict[i + 1]
                    writer.add_scalar(f"{phase}/{key}/{class_name}", val.item(), epoch)
            else:
                writer.add_scalar(f"{phase}/{key}", value.item(), epoch)
