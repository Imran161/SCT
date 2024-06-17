import torch

from utils import iou_metric, SCT_out_classes, ImageVisualizer
from metrics import DetectionMetrics


def test_model(
    model,
    model_weight,
    criterion,
    train_loader,
    train_predict_path,
    val_loader,
    val_predict_path,
    device,
    num_classes,
    epoch=None,
):
    train_image_visualizer = ImageVisualizer(train_predict_path)
    val_image_visualizer = ImageVisualizer(val_predict_path)

    # writer = SummaryWriter(log_dir="weak_logs")
    metrics_calculator = DetectionMetrics(mode="ML", num_classes=num_classes)

    model.load_state_dict(torch.load(model_weight))
    model.to(device)
    model.eval()  # Перевод модели в режим оценки

    # class_names_dict = {class_info['id']: class_info['name'] for class_info in SCT_out_classes}
    # class_names_dict = {
    #     1: "Внутримозговое кровозлияние",
    #     2: "Субарахноидальное кровозлияние",
    #     3: "Cубдуральное кровозлияние",
    #     4: "Эпидуральное кровозлияние",
    # }
    
    class_names_dict = {
        1: "Снижение пневматизации околоносовых пазух",
        2: "Горизонтальный уровень жидкость-воздух"
    }

    # Инициализация переменных для метрик
    val_loss_sum = 0.0
    val_iou_sum = torch.zeros(num_classes)

    colors = [
        ((251, 206, 177), "Абрикосовым"),
        ((127, 255, 212), "Аквамариновым"),
        ((255, 36, 0), "Алым"),
        ((153, 102, 204), "Аметистовым"),
        ((153, 0, 102), "Баклажановым"),
        ((48, 213, 200), "Бирюзовым"),
        ((152, 251, 152), "Бледно зеленым"),
        ((213, 113, 63), "Ванильным"),
        ((100, 149, 237), "Васильковым"),
        ((34, 139, 34), "Зелёный лесной"),
        ((0, 0, 255), "Синий"),
        ((75, 0, 130), "Индиго"),
        ((255, 0, 255), "Чёрный"),
        ((0, 51, 153), "Маджента"),
        ((65, 105, 225), "Королевский синий"),
        ((255, 255, 0), "Жёлтый"),
        ((255, 69, 0), "Оранжево-красный"),
        ((255, 0, 0), "Темно синим"),
        ((0, 51, 153), "Красный"),
        ((255, 215, 0), "Золотой"),
        ((250, 128, 114), "Лососевый"),
        ((255, 99, 71), "Томатный"),
        ((255, 215, 0), "Золотой"),
        ((0, 139, 139), "Тёмный циан"),
        ((0, 255, 255), "Морская волна"),
    ]

    with torch.no_grad():
        for train_batch in train_loader:
            # optimizer.zero_grad()
            images = train_batch["images"].to(device)
            masks = train_batch["masks"][:, 1:, :, :].to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)

            # loss = criterion(outputs, masks)
            # loss.backward()
            # optimizer.step()
            # train_loss_sum += loss.item()

            # train_iou_batch = iou_metric(outputs, masks, num_classes)
            # # train_iou_sum += torch.sum(train_iou_batch, dim=0)  # Суммирование IoU для каждого класса по всем батчам
            # # вроде так
            # train_iou_sum += train_iou_batch

            # # для трейна метрики тоже посчитаю
            # metrics_calculator.update_counter(masks, outputs)

            train_image_visualizer.visualize(
                images, masks, outputs, class_names_dict, colors, epoch
            )

        for k, val_batch in enumerate(val_loader):
            images_val = val_batch["images"].to(device)
            masks_val = val_batch["masks"][:, 1:].to(device)
            outputs_val = model(images_val)

            outputs_val = torch.sigmoid(outputs_val)

            # лоссы я убрал
            # val_loss_sum += criterion(outputs_val, masks_val).item()
            # val_iou_batch = iou_metric(outputs_val, masks_val, num_classes)
            # val_iou_sum += val_iou_batch
            # metrics_calculator.update_counter(masks_val, outputs_val)

            val_image_visualizer.visualize(
                images_val, masks_val, outputs_val, class_names_dict, colors, epoch
            )

        # тут все норм но я в даталоудеры хочу 32 элемента передать и тогда тут ошибка
        try:
            val_loss_avg = val_loss_sum / len(val_loader)
            val_iou_avg = val_iou_sum / len(val_loader)

            # print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss_avg}, Val Loss: {val_loss_avg},  Val IoU: {val_iou_avg}")
            print(f"Val Loss: {val_loss_avg},  Val IoU: {val_iou_avg}")
        except:
            pass
