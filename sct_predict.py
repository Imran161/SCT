import torch
from torch.utils.data import DataLoader
from your_dataset_module import YourDatasetClass  # Замените на имя вашего модуля с классом для загрузки данных
from train_sct import iou_metric, combined_loss  # Замените на имя вашего модуля с метриками и функцией для расчета combined loss
import segmentation_models_pytorch as smp

# Функция для оценки модели на валидационной выборке
def evaluate_model(model_path, val_loader, device, num_classes):
    # Загрузка модели
    model = smp.FPN(encoder_name="efficientnet-b7", encoder_weights="imagenet", in_channels=1, classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Определение критерия оценки
    criterion = combined_loss  # Замените на вашу функцию оценки

    # Оценка модели
    val_loss = 0.0
    val_iou_sum = torch.zeros(num_classes)
    num_samples = len(val_loader.dataset)

    with torch.no_grad():
        for val_batch in val_loader:
            images_val = val_batch["images"].to(device)
            masks_val = val_batch["masks"][:, 1:].to(device)  # Исключаем фоновый класс
            outputs_val = model(images_val)
            outputs_val = torch.sigmoid(outputs_val)

            val_loss += criterion(outputs_val, masks_val).item()

            val_iou_batch = iou_metric(outputs_val, masks_val, num_classes)
            val_iou_sum += val_iou_batch

    val_loss_avg = val_loss / len(val_loader)
    val_iou_avg = val_iou_sum / num_samples

    print(f"Validation Loss: {val_loss_avg}, Validation IoU: {val_iou_avg}")


if __name__ == "__main__":
    # Пути к модели и валидационным данным
    model_path = "best_segmentation_model.pth"  # Путь к сохраненным весам модели
    val_data_path = "/path/to/your/validation/data"  # Путь к вашим валидационным данным

    # Загрузка валидационных данных
    val_dataset = YourDatasetClass(val_data_path)  # Инициализация вашего класса для загрузки данных
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Оценка модели на валидационной выборке
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 4  # Замените на количество классов вашей задачи
    evaluate_model(model_path, val_loader, device, num_classes)
