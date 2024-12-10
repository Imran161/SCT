import torch
import os


def standart_weight_saving_manager(variables, model, config):
    """
    Сохраняет текущую модель и модель с лучшей ошибкой на валидации.
    """
    save_dir = os.path.join(config["model_save_dir"], "models")
    os.makedirs(save_dir, exist_ok=True)

    current_model_path = os.path.join(save_dir, f"model_epoch_{variables['current_epoch']}.pth")
    torch.save(model.state_dict(), current_model_path)
    print(f"Сохранена текущая модель на эпохе {variables['current_epoch']}")

    if "best_val_loss" not in variables or variables["current_loss"] < variables["best_val_loss"]:
        variables["best_val_loss"] = variables["current_loss"]
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
                print("Validation phase: no_grad active")
                return func(self, *args, **kwargs)
        print("Training phase: gradients enabled")
        return func(self, *args, **kwargs)
    return wrapper


def standart_logging_manager(config, experiment_name):
    from torch.utils.tensorboard import SummaryWriter
    log_dir = os.path.join(config["model_save_dir"], f"{experiment_name}_logs")
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)


def log_metrics(writer, phase, metrics, epoch):
    """
    Логирует метрики в TensorBoard.
    """
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if len(value.size()) > 0:
                writer.add_scalar(f"{phase}/Mean/{key}", value.mean().item(), epoch)
                for i, val in enumerate(value):
                    class_name = f"Class_{i+1}"
                    writer.add_scalar(f"{phase}/{key}/{class_name}", val.item(), epoch)
            else:
                writer.add_scalar(f"{phase}/{key}", value.item(), epoch)
