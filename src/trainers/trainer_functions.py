import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from ..losses.losses_cls import (
    FocalLoss,
    WeakCombinedLoss,
)  # Подключи кастомные функции лосса
from ..metrics.metrics import DetectionMetrics

writer = SummaryWriter()


def standart_model_configurate(model, variables):
    if variables["current_phase"] == "train":
        model.train()
    elif variables["current_phase"] == "val":
        model.eval()


def apply_activation(outputs, config):
    activation_fn = config.get("activation", None)
    if activation_fn == "sigmoid":
        return torch.sigmoid(outputs)
    elif activation_fn == "softmax":
        return F.softmax(outputs)
    return outputs


def standart_batch_function(batch):
    return {"images": batch["images"], "masks": batch["mask"][:, 1:, :, :]}


def standart_logging_manager(variables, functions):
    """
    Логирование метрик в TensorBoard.
    Ожидает, что в `functions` определена функция `compute_metrics` для подсчета метрик.
    """
    if functions["metrick"]:
        phase = variables["current_phase"]
        epoch = variables["current_epoch"]

        metrics = functions["metrick"].compute_metrics()
        for metric_name, value in metrics.items():
            writer.add_scalar(f"{phase}/{metric_name}", value, epoch)

        functions["metrick"].reset()


def standart_weight_saving_manager(variables, model, config):
    """
    Сохранение весов модели, если текущая ошибка ниже наилучшей.
    Ожидает, что `variables` содержит значения 'epoch_loss' и 'Best loss',
    а `config` - путь для сохранения модели в 'save_path'.
    """
    # Инициализация наилучшей ошибки, если не задана
    if "Best loss" not in variables:
        variables["Best loss"] = float("inf")

    current_loss = variables["epoch_loss"]

    # Проверка, является ли текущая ошибка лучше наилучшей ошибки
    if current_loss < variables["Best loss"]:
        variables["Best loss"] = current_loss

        # Сохранение модели
        save_path = config.get("save_path", "best_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Модель сохранена с лучшей ошибкой: {current_loss} на пути: {save_path}")
