import itertools
import os
import warnings

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from ..metrics.metrics import DetectionMetrics
from sklearn.exceptions import UndefinedMetricWarning
from torch.optim import Adam
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..datamanager.coco_classes import (
    kidneys_base_classes,
    kidneys_pat_out_classes,
)
from ..datamanager.coco_dataloaders import SINUSITE_COCODataLoader
from ..transforms.transforms import SegTransform
from ..utils.inference import test_model
from ..utils.seed import set_seed
from ..utils.utils import (
    ExperimentSetup,
    iou_metric,
    save_best_metrics_to_csv,
)

мне нужно будет потом сделать segmentation trainer, regression, detection, diffusion, gan,
multimodal transformer(florence-2), multioutput(вход обычный, а выход любой может быть)

set_seed(64)


def get_direct_subdirectories(directory):
    subdirectories = [
        d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))
    ]
    return [os.path.join(directory, subdir) for subdir in subdirectories]


def custom_collate_fn(batch):
    images = []
    masks = []
    for item in batch:
        image, mask = item["images"], item["masks"]
        images.append(image)
        masks.append(mask)

    collated_images = default_collate(images)
    collated_masks = default_collate(masks)

    return {"images": collated_images, "masks": collated_masks}


def add_noise_to_mask(mask, max_m=3, max_n=3):
    mask = mask.float()  # Преобразуем маску к типу float
    noise = torch.rand_like(mask)  # Генерируем равномерный шум на [0, 1]
    m = torch.randint(
        0, max_m + 1, (1,)
    ).item()  # Генерируем целое число от 1 до max_m включительно
    n = torch.randint(
        0, max_n + 1, (1,)
    ).item()  # Генерируем целое число от 0 до max_n включительно

    if m == 0 and n == 0:
        if torch.rand(1) < 0.5:  # С вероятностью 50% выбираем между m и n
            m = torch.randint(1, max_m + 1, (1,)).item()
        else:
            n = torch.randint(1, max_n + 1, (1,)).item()

    SMOOTH = 1e-8
    noisy_mask = (m * mask + n * noise) / (m + n + SMOOTH)
    return noisy_mask


def add_noise(images, masks, x):
    # images размера [batch_size, num_channels, height, width]
    # masks размера [batch_size, num_masks, height, width]
    batch_size, num_masks, height, width = masks.shape
    num_channels = images.shape[1]

    # Создаем пустой тензор для склеенных данных
    combined = torch.empty(
        batch_size, num_channels + num_masks, height, width, device=images.device
    )

    for i in range(batch_size):
        noise = torch.rand(num_masks, height, width, device=images.device)
        #         X = torch.rand(1, device=images.device)
        noisy_masks = noise
        noisy_masks = x * masks[i] + (1 - x) * noise
        combined[i] = torch.cat([images[i], noisy_masks], dim=0)

    return combined


def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-8)
    return normalized_tensor


def add_noise_and_combine(
    images, masks, epoch, num_epochs, k_values, batch_idx, num_batches
):
    with torch.no_grad():
        batch_size, num_masks, height, width = masks.shape
        num_channels = images.shape[1]

        # проверил, рисует нормально
        # for k in k_values:
        #     # Создаем пустой тензор для склеенных данных
        #     combined = torch.empty(batch_size, num_channels + num_masks, height, width, device=images.device)

        #     # Определяем уровень шума
        #     noise_level = epoch / num_epochs
        #     num_noisy_masks = int(batch_size * (0.5 + 0.4 * noise_level))

        #     for i in range(batch_size):
        #         # Создаем гауссовский шум и клэмпим его
        #         # noise = torch.randn(num_masks, height, width, device=images.device)
        #         noise = torch.normal(0, k, size=(num_masks, height, width), device=images.device)

        #         # Определяем, будет ли маска сильно зашумлена или нет
        #         if i < num_noisy_masks:
        #             noisy_masks = (masks[i] + noise).clamp(0, 1) # torch.clamp(masks[i] + noise, 0, 1)
        #         else:
        #             noisy_masks = masks[i]

        #         combined[i] = torch.cat([images[i], noisy_masks], dim=0)

        #     combined_np = combined[0][1].cpu().numpy()
        #     # print("combined.shape", combined.shape)
        #     img_combined = (combined_np * 255).astype(np.uint8)
        #     output_dir = "/home/imran-nasyrov/noisy_masks"
        #     filename = f"{output_dir}/combined_k{k}.jpg"
        #     cv2.imwrite(filename, img_combined)
        #     print(f"Saved {filename}")

        # Определяем значение k в зависимости от текущей эпохи и индекса батча
        if epoch == 0:
            if batch_idx < num_batches / 2:
                k = 8
            else:
                k = (batch_idx - num_batches / 2) / (num_batches / 2) * 8
        else:
            percent = 0.5 + (epoch / num_epochs) * 0.4  # 0.5 -> 0.9 over epochs
            if batch_idx < num_batches * percent:
                k = 8
            else:
                k = (
                    8
                    + (batch_idx - num_batches * percent)
                    / (num_batches * (1 - percent))
                    * 8
                )

        # Создаем пустой тензор для склеенных данных
        combined = torch.empty(
            batch_size, num_channels + num_masks, height, width, device=images.device
        )
        num_noisy_masks = int(batch_size * (0.5 + 0.4 * (epoch / num_epochs)))

        for i in range(batch_size):
            # Создаем гауссовский шум с заданным стандартным отклонением k
            noise = torch.normal(
                0, k, size=(num_masks, height, width), device=images.device
            )

            # Определяем, будет ли маска сильно зашумлена или нет
            if i < num_noisy_masks:
                noisy_masks = (masks[i] + noise).clamp(0, 1)
            else:
                noisy_masks = masks[i]

            combined[i] = torch.cat([images[i], noisy_masks], dim=0)

    return combined.clone()


def save_images(images, noisy_masks, epoch, batch_idx, save_dir="noisy_masks"):
    path = f"{save_dir}/{epoch}/{batch_idx}"
    os.makedirs(path, exist_ok=True)
    batch_size = images.size(0)

    # for i in range(batch_size):
    #     # Преобразуем изображения и маски к формату (H, W, C) и диапазону [0, 255]
    #     img = (images[i].cpu().numpy() * 255).astype(np.uint8)#.squeeze(0)
    #     mask_0 = (noisy_masks[i][0].cpu().numpy() * 255).astype(np.uint8)
    #     mask_1 = (noisy_masks[i][1].cpu().numpy() * 255).astype(np.uint8)
    #
    #     # Сохраняем изображения
    #     cv2.imwrite(os.path.join(path, f"epoch_{epoch+1}_batch_{batch_idx}_img_{i+1}.jpg"), img)
    #     cv2.imwrite(os.path.join(path, f"epoch_{epoch+1}_batch_{batch_idx}_mask0_{i+1}.jpg"), mask_0)
    #     cv2.imwrite(os.path.join(path, f"epoch_{epoch+1}_batch_{batch_idx}_mask1_{i+1}.jpg"), mask_1)

    for i in range(batch_size):
        # Преобразуем изображения и маски к формату (H, W, C) и диапазону [0, 255]
        img = (images[i].cpu().numpy() * 255).astype(np.uint8).squeeze(0)
        mask_0 = (noisy_masks[i][0].cpu().numpy() * 255).astype(np.uint8)
        mask_1 = (noisy_masks[i][1].cpu().numpy() * 255).astype(np.uint8)

        # Объединяем изображения горизонтально
        combined_image = np.hstack((img, mask_0, mask_1))

        # Сохраняем комбинированное изображение
        cv2.imwrite(
            os.path.join(path, f"epoch_{epoch}_batch_{batch_idx}_img_{i}.jpg"),
            combined_image,
        )


def train_model(
    model,
    optimizer,
    criterion,
    lr_sched,
    num_epochs,
    train_loader,
    val_loader,
    device,
    num_classes,
    experiment_name,
    all_class_weights,
    alpha,
    use_opt_pixel_weight,
    num_cyclic_steps,  # количество циклических шагов на валидации
    max_n=3,
    max_k=3,
    use_augmentation=False,
    loss_type="weak",
):
    # Создание объекта SummaryWriter для записи логов
    # writer = SummaryWriter(log_dir=f"runs_sinusite/{experiment_name}_logs")
    writer = SummaryWriter(log_dir=f"runs_kidneys/{experiment_name}_logs")
    # metrics_calculator = DetectionMetrics(mode="ML", num_classes=num_classes)

    # Создаем два объекта для подсчета метрик
    metrics_calculator_train = DetectionMetrics(mode="ML", num_classes=num_classes)
    metrics_calculator_val = DetectionMetrics(mode="ML", num_classes=num_classes)

    class_names_dict = {
        class_info["id"]: class_info["name"]
        # sinusite_pat_classes_3 или kidneys_pat_out_classes
        for class_info in kidneys_pat_out_classes
    }
    print("class_names_dict", class_names_dict)

    classes = list(class_names_dict.keys())
    weight_opt = Weight_opt_class(criterion, classes, None)

    # print("device", device)
    model = model.to(device)

    best_loss = 100

    global_stats = {
        "global_loss_sum": torch.tensor(0.0, dtype=torch.double),
        "global_loss_numel": torch.tensor(0.0, dtype=torch.double),
    }

    if alpha is not None:
        alpha_no_fon = np.array([arr[1:] for arr in alpha])
        # alpha_no_fon = np.array(alpha[1:], dtype=np.float16) было так вместо верхней строки
        alpha_no_fon = torch.tensor(alpha_no_fon).to(device)
    else:
        alpha_no_fon = None

    if use_augmentation:
        seg_transform = SegTransform()

    num_batches = len(train_loader)

    for epoch in range(num_epochs):
        # убрал
        # torch.cuda.empty_cache() # должно освобождать память
        # print("gpu_usage()", gpu_usage())

        model.train()
        train_loss_sum = 0.0
        val_loss_sum = 0.0  # сюда переместил
        # train_iou_sum = 0.0
        # val_iou_sum = 0.0
        train_iou_sum = torch.zeros(num_classes)
        val_iou_sum = torch.zeros(num_classes)

        n = 0
        ############################
        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
        ) as pbar:
            for batch_idx, train_batch in enumerate(train_loader):
                optimizer.zero_grad()

                # images, masks = train_batch
                # # print("masks 0", masks[0][2]) тут по ходу не фон нулевой, там не только единицы
                # masks = masks[:, 1:, :, :].to(device)
                # Используем только пиксельные значения
                # pixel_values = images["pixel_values"].to(device)
                # input_ids = images["input_ids"].to(device)
                # print("input_ids shape", input_ids.shape)
                # print("pixel_values", pixel_values)

                images = train_batch["images"].to(device)
                masks = train_batch["masks"][:, 1:, :, :].to(device)

                # print("images shape", images.shape)
                # print("masks shape", masks.shape)
                # unique_values = torch.unique(masks)
                # num_unique_values = unique_values.numel()
                # print("Уникальные значения:", unique_values)
                # print("Количество уникальных значений:", num_unique_values)

                if use_augmentation:
                    images, masks = seg_transform.apply_transform(images, masks)
                    # save_images(images, masks, epoch, batch_idx, save_dir="transforms")

                # шум
                # шум к маске
                # k_values = np.arange(0, 10.1, 0.1)
                # combined = add_noise_and_combine(
                #     images, masks, epoch, num_epochs, k_values, batch_idx, num_batches
                # )

                if all_class_weights is not None:
                    all_weights_no_fon = [x[1:] for x in all_class_weights]
                else:
                    all_weights_no_fon = None

                # тут n сделаю
                # n += 1

                # шум
                # outputs = model(
                #     combined
                # )  # 2 канала на выходе 3 на входе (num_classes + 1)
                # outputs = torch.tanh(outputs)
                # outputs = torch.sigmoid(outputs)

                # Вычитаем предсказанный шум из исходного изображения
                # outputs = (combined[:, 1:, :, :] - outputs + 1) / 3.0
                # loss = criterion(outputs, masks, all_weights_no_fon, alpha_no_fon)

                # не шум
                outputs = model(images)
                # outputs = model(pixel_values=images)
                outputs = torch.sigmoid(outputs)

                if loss_type == "weak" or loss_type == "strong":
                    loss = criterion(outputs, masks, all_class_weights, alpha_no_fon)
                elif loss_type == "focus":
                    loss, global_loss_sum, global_loss_numel = criterion(
                        outputs,
                        masks,
                        global_loss_sum=global_stats["global_loss_sum"],
                        global_loss_numel=global_stats["global_loss_numel"],
                        train_mode=True,
                        mode="ML",
                    )

                    #####################################
                    # это не надо уже, в самом лоссе все считается
                    # global_stats["global_loss_sum"] = global_loss_sum  # / n
                    # global_stats["global_loss_numel"] = global_loss_numel  # / n

                    # print("global_stats", global_stats)

                elif loss_type == "bce":
                    loss = criterion(outputs, masks)

                # было так но я добавил focus loss выше
                # loss = criterion(outputs, masks, all_weights_no_fon, alpha_no_fon)

                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()

                # Сохранение изображений и масок
                # save_images(images, combined[:, 1:, :, :], epoch, batch_idx)

                # шум
                # train_iou_batch = iou_metric(outputs, masks, num_classes)
                # не шум
                train_iou_batch = iou_metric(outputs, masks, num_classes)
                train_iou_sum += train_iou_batch

                # для трейна метрики тоже посчитаю
                metrics_calculator_train.update_counter(
                    masks,
                    outputs,  # outputs не шум
                )  # , advanced_metrics=True)

                # скользящее среднее
                n += 1
                pbar.set_postfix(loss=train_loss_sum / n)
                pbar.update(1)

                # оптимизация весов

        train_loss_avg = train_loss_sum / len(train_loader)

        # среднее по всем батчам
        # train_iou_avg = train_iou_sum / len(train_loader)
        train_iou_avg = train_iou_sum / len(train_loader)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            train_metrics = metrics_calculator_train.calc_metrics()

        # было так но я написал ниже, там для каждого класса метрика
        # for key, value in train_metrics.items():
        #     if isinstance(value, torch.Tensor):
        #         writer.add_scalar(f"Train/{key}", value.mean().item(), epoch) # вот так было а я сделал две строчки внизу
        #     elif isinstance(value, list):
        #         for i, val in enumerate(value):
        #             writer.add_scalar(f"Train/{key}/Class_{i}", val.item(), epoch)

        for key, value in train_metrics.items():
            if isinstance(value, torch.Tensor):
                if len(value.size()) > 0:  # Проверяем, что тензор не пустой
                    # средняя метрика
                    writer.add_scalar(f"Train/Mean/{key}", value.mean().item(), epoch)
                    for i, val in enumerate(value):
                        class_name = class_names_dict[i + 1]
                        # writer.add_scalar(f"Train/{key}/Class_{i}", val.item(), epoch)
                        writer.add_scalar(
                            f"Train/{key}/{class_name}", val.item(), epoch
                        )
                else:
                    writer.add_scalar(f"Train/{key}", value.item(), epoch)

        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

        print("alpha_no_fon", alpha_no_fon)
        if alpha_no_fon is not None:
            # print("class", class_names_dict[1])
            # print("alpha_no_fon pixel_pos_weights", alpha_no_fon[0])

            print(
                f"\nclass: {class_names_dict[1]}, pixel_pos_weights {alpha_no_fon[0][0]}"
            )
            print(
                f"class: {class_names_dict[2]}, pixel_pos_weights {alpha_no_fon[0][1]}"
            )
            print(
                f"class: {class_names_dict[3]}, pixel_pos_weights {alpha_no_fon[0][2]}\n"
            )

            print(
                f"class: {class_names_dict[1]}, pixel_neg_weights {alpha_no_fon[1][0]}"
            )
            print(
                f"class: {class_names_dict[2]}, pixel_neg_weights {alpha_no_fon[1][1]}"
            )
            print(
                f"class: {class_names_dict[3]}, pixel_neg_weights {alpha_no_fon[1][2]}\n"
            )

            print(
                f"class: {class_names_dict[1]}, pixel_class_weights {alpha_no_fon[2][0]}"
            )
            print(
                f"class: {class_names_dict[2]}, pixel_class_weights {alpha_no_fon[2][1]}"
            )
            print(
                f"class: {class_names_dict[3]}, pixel_class_weights {alpha_no_fon[2][2]}\n"
            )

            # print("class", class_names_dict[2])
            # print("alpha_no_fon pixel_neg_weights", alpha_no_fon[1])
            # print("class", class_names_dict[3])
            # print("alpha_no_fon pixel_class_weights", alpha_no_fon[2])

        print("class_names_dict", class_names_dict)

        # оптимизация пиксельных весов
        if use_opt_pixel_weight:
            alpha_no_fon = weight_opt.opt_pixel_weight(train_metrics, alpha_no_fon)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss_avg}, Train IoU: {train_iou_avg}"
        )
        ###############################################################
        # Валидация
        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                images_val = val_batch["images"].to(device)
                masks_val = (
                    val_batch["masks"][:, 1:].to(device)  # .float()
                )  # float() для шума добавил

                # шум
                # noise_val = torch.rand_like(masks_val).to(device)
                # combined_val = torch.cat((images_val, noise_val), dim=1)

                # outputs_val = model(combined_val)
                # outputs_val = torch.tanh(outputs_val)
                # # outputs_val = torch.sigmoid(outputs_val)
                # corrected_masks_val = outputs_val
                # corrected_masks_val = (combined_val[:, 1:, :, :] - outputs_val + 1) / 3
                # # # Циклический процесс для предсказаний
                # # # сделать ноль этого цикла, на тесте сделаем цикл
                # # # for _ in range(num_cyclic_steps):
                # # #     outputs_val = model(images_val)
                # # #     outputs_val = torch.sigmoid(outputs_val)
                # val_loss_sum += criterion(
                #     corrected_masks_val, masks_val, None, None
                # ).item()
                # val_iou_batch = iou_metric(corrected_masks_val, masks_val, num_classes)

                # не шум
                outputs_val = model(images_val)
                outputs_val = torch.sigmoid(outputs_val)

                if loss_type == "weak" or loss_type == "strong":
                    val_loss_sum += criterion(outputs_val, masks_val, None, None).item()

                elif loss_type == "focus":
                    criterion.reset_global_loss()  # Надо метод в валидации делать reset_global_loss
                    val_loss, _, _ = criterion(
                        outputs_val,
                        masks_val,
                        global_loss_sum=None,
                        global_loss_numel=None,
                        train_mode=False,
                        mode="ML",
                    )
                    val_loss_sum += val_loss.item()

                elif loss_type == "bce":
                    val_loss_sum += criterion(outputs_val, masks_val).item()

                # было так но я добавил focus loss выше
                # val_loss_sum += criterion(outputs_val, masks_val, None, None).item()

                val_iou_batch = iou_metric(outputs_val, masks_val, num_classes)

                val_iou_sum += val_iou_batch

                metrics_calculator_val.update_counter(
                    masks_val,
                    outputs_val,  # не шум
                    # corrected_masks_val,  # шум
                )  # advanced_metrics=True)

                # val_image_visualizer.visualize(images_val, masks_val, outputs_val, class_names_dict, colors, epoch)

            val_loss_avg = val_loss_sum / len(val_loader)

            # val_iou_avg = val_iou_sum / len(val_loader)
            val_iou_avg = val_iou_sum / len(val_loader)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss_avg}, Val Loss: {val_loss_avg},  Val IoU: {val_iou_avg}"
        )

        if lr_sched is not None:
            lr_sched.step()  # когда мы делаем эту команду он залезает в optimizer и изменяет lr умножая его на 0.5

        # обработаю исключение но не знаю хорошая идея или нет
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            val_metrics = metrics_calculator_val.calc_metrics()

        for key, value in val_metrics.items():
            if isinstance(value, torch.Tensor):
                if len(value.size()) > 0:
                    # добавил среднюю метрику по классам
                    writer.add_scalar(f"Val/Mean/{key}", value.mean().item(), epoch)
                    for i, val in enumerate(value):
                        class_name = class_names_dict[i + 1]
                        # writer.add_scalar(f"Val/{key}/Class_{i}", val.item(), epoch)
                        writer.add_scalar(f"Val/{key}/{class_name}", val.item(), epoch)
                else:
                    writer.add_scalar(f"Val/{key}", value.item(), epoch)

        for class_idx, iou_value in enumerate(train_iou_avg):
            class_name = class_names_dict[
                class_idx + 1
            ]  # в out classes индексы с единицы
            writer.add_scalar(f"My_train_IoU/{class_name}", iou_value, epoch)

        for class_idx, iou_value in enumerate(val_iou_avg):
            class_name = class_names_dict[class_idx + 1]
            writer.add_scalar(f"My_val_IoU/{class_name}", iou_value, epoch)

        writer.add_scalar("Loss/train", train_loss_avg, epoch)
        writer.add_scalar("Loss/validation", val_loss_avg, epoch)

        # тут сохранение лучшей модели
        if val_loss_avg < best_loss:
            best_loss = val_loss_avg

            # Сохранение метрик в CSV
            best_metrics = {
                "experiment": experiment_name.split("_")[0],
                "epoch": epoch,
                "train_loss": train_loss_avg,
                "val_loss": val_loss_avg,
                "val_metrics": {
                    "IOU": val_metrics["IOU"],
                    "F1": val_metrics["F1"],
                    "area_probs_F1": val_metrics["area_probs_F1"],
                },
            }

            # best_model_path = "sinusite_best_models"
            best_model_path = "kidneys_best_models"
            if not os.path.exists(best_model_path):
                os.makedirs(best_model_path)

            torch.save(
                model.state_dict(),
                f"{best_model_path}/best_{experiment_name}_model.pth",
            )

            csv_file = f"{best_model_path}/best_metrics.csv"
            save_best_metrics_to_csv(best_metrics, csv_file)

    last_metrics = {
        "experiment": experiment_name.split("_")[0],
        "epoch": epoch,
        "train_loss": train_loss_avg,
        "val_loss": val_loss_avg,
        "val_metrics": {
            "IOU": val_metrics["IOU"],
            "F1": val_metrics["F1"],
            "area_probs_F1": val_metrics["area_probs_F1"],
        },
    }

    # last_model_path = "sinusite_last_models"
    last_model_path = "kidneys_last_models"
    if not os.path.exists(last_model_path):
        os.makedirs(last_model_path)

    torch.save(
        model.state_dict(),
        f"{last_model_path}/last_{experiment_name}_model.pth",
    )

    last_csv_file = f"{last_model_path}/last_metrics.csv"
    save_best_metrics_to_csv(last_metrics, last_csv_file)

    writer.close()


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
    # path = "/home/imran-nasyrov/sinusite_json_data"
    # subdirectories_list = get_direct_subdirectories(path)

    batch_size = 24
    num_classes = 3

    # sinusite
    # params = {
    #     "json_file_path": "/home/imran-nasyrov/sinusite_json_data",
    #     "delete_list": [],
    #     "base_classes": sinusite_base_classes,
    #     "out_classes": sinusite_pat_classes_3,
    #     "dataloader": True,
    #     "resize": (1024, 1024),
    #     "recalculate": False,
    #     "delete_null": False,
    # }

    # kidneys
    params = {
        "json_file_path": "/home/imran-nasyrov/json_pochki",
        "delete_list": [],
        "base_classes": kidneys_base_classes,
        # kidneys_out_classes или kidneys_pat_out_classes
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

    # for batch_idx, train_batch in enumerate(train_loader):
    #     print(f"Batch {batch_idx}:")
    #     for i, item in enumerate(train_batch["images"]):
    #         print(f"  Item {i}: image shape = {item.shape}")
    #     for i, item in enumerate(train_batch["masks"]):
    #         print(f"  Item {i}: mask shape = {item.shape}")

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
        classes=num_classes,
    )

    # # segformer
    # config = SegformerConfig.from_pretrained(
    #     "nvidia/segformer-b0-finetuned-ade-512-512",
    #     num_channels=1,
    #     num_labels=2,
    #     # ignore_mismatched_sizes=True, можно попробовать это сделать
    # )
    # # config.num_channels = 1  # Изменение на одноканальные изображения
    # # config.num_labels = 2  # Изменение на два класса
    #
    # # Создание новой модели с этой конфигурацией
    # model = SegformerForSemanticSegmentation(config)
    #
    # class SegformerForSemanticSegmentation(nn.Module):
    #     def __init__(self, model, output_size=(1024, 1024)):
    #         super(SegformerForSemanticSegmentation, self).__init__()
    #         self.model = model
    #         self.output_size = output_size
    #
    #     def forward(self, x):
    #         # print("x.shape", x.shape)
    #         # Преобразование одноканального изображения в трехканальное
    #         # x = x.repeat(1, 3, 1, 1)
    #         outputs = self.model(pixel_values=x)
    #
    #         # print("Type of outputs:", type(outputs))
    #         # print("Keys in outputs:", outputs.keys())
    #         # print("outputs.logits shape", outputs.logits.shape)
    #
    #         logits = outputs.logits
    #         # print("logits shape", logits.shape)
    #         logits = F.interpolate(
    #             logits, size=self.output_size, mode="bilinear", align_corners=False
    #         )
    #         # print("logits shape after interpolation", logits.shape)
    #
    #         return logits
    #
    # model = SegformerForSemanticSegmentation(model).to(device)
    #
    # print("model", model)

    # пробую florence-2, пишет что cuda должна быть версии 11.6 и выше, у нас 11.5
    # model_id = 'microsoft/Florence-2-large'
    # model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    # processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # # Пример настройки модели для одноканальных изображений
    # class FlorenceSegmentationModel(nn.Module):
    #     def __init__(self, model):
    #         super(FlorenceSegmentationModel, self).__init__()
    #         self.model = model

    #     def forward(self, x, input_ids=None, decoder_input_ids=None, decoder_inputs_embeds=None):
    #         print("x shape", x.shape)
    #         # Преобразование одноканального изображения в трехканальное
    #         x = x.repeat(1, 3, 1, 1)
    #         # input_ids = torch.randint(0, 100, (x.size(0), 1025)).to(device)
    #         input_ids = torch.randint(0, 100, (1, 10)).to(device)
    #         print("input_ids shape", input_ids.shape)
    #         outputs = self.model(pixel_values=x, input_ids=input_ids, decoder_input_ids=decoder_input_ids, decoder_inputs_embeds=decoder_inputs_embeds)

    #         # # print("Type of outputs:", type(outputs))
    #         # # print("Keys in outputs:", outputs.keys())
    #         # print("outputs.logits shape", outputs.logits.shape)
    #         # return outputs.logits

    #         print("Type of outputs:", type(outputs))
    #         print("Keys in outputs:", outputs.keys())
    #         if 'logits' in outputs:
    #             print("outputs.logits shape", outputs.logits.shape)
    #         if 'image_features' in outputs:
    #             print("outputs.image_features shape", outputs.image_features.shape)
    #         if 'task_prefix_embeds' in outputs:
    #             print("outputs.task_prefix_embeds shape", outputs.task_prefix_embeds.shape)
    #         return outputs

    # model = FlorenceSegmentationModel(model).to(device)

    learning_rate = 3e-4
    num_epochs = 120

    optimizer = Adam(model.parameters(), lr=learning_rate)
    lr_sched = None

    use_class_weight = False
    use_pixel_weight = False
    use_pixel_opt = False
    power = "1.6.2_test_new_code_kidneys_weak"  # focus или weak

    loss_type = power.split("_")[-1]
    print("loss_type", loss_type)

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

    print("experiment_name", experiment_name)
    print("criterion", criterion)

    train_model(
        model,
        optimizer,
        criterion,
        lr_sched,
        num_epochs,
        train_loader,
        val_loader,
        device,
        num_classes,
        experiment_name,
        all_class_weights=all_class_weights,
        alpha=pixel_all_class_weights,
        use_opt_pixel_weight=use_pixel_opt,
        num_cyclic_steps=0,
        max_n=3,
        max_k=3,
        use_augmentation=False,
        loss_type=loss_type,
    )

    model_weight = f"sinusite_best_models/best_{experiment_name}_model.pth"

    val_predict_path = f"diff_predict_sinusite/predict_{experiment_name}/val"
    train_predict_path = f"diff_predict_sinusite/predict_{experiment_name}/train"

    limited_train_loader = itertools.islice(train_loader, 6)
    limited_val_loader = itertools.islice(val_loader, 6)

    # avg_loss = test_model(
    #     model,
    #     model_weight,
    #     criterion,
    #     train_loader,# limited_train_loader,
    #     train_predict_path,
    #     val_loader, #limited_val_loader,
    #     val_predict_path,
    #     device,
    #     num_classes,
    #     num_images_to_draw=36
    # )
