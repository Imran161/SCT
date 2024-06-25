import itertools
import os
import csv
import warnings

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import torch.nn as nn
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerFeatureExtractor,
    SegformerConfig,
    AutoImageProcessor,
    AutoModelForImageSegmentation,
    AutoProcessor,
    AutoModelForCausalLM,
)
import torch
from sklearn.exceptions import UndefinedMetricWarning
from torch.optim import Adam
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from coco_classes import sinusite_base_classes, sinusite_pat_classes_3
from coco_dataloaders import SINUSITE_COCODataLoader
from metrics import DetectionMetrics
from sct_val import test_model
from transforms import SegTransform
from utils import ExperimentSetup, iou_metric, set_seed

set_seed(64)


def save_best_metrics_to_csv(best_metrics, csv_file):
    # Проверка, существует ли CSV файл
    file_exists = os.path.isfile(csv_file)

    # Если файл существует, считываем данные
    if file_exists:
        with open(csv_file, mode="r", newline="") as file:
            reader = csv.reader(file)
            data = list(reader)
        headers = data[0]
        rows = data[1:]
    else:
        # Если файл не существует, инициализируем заголовки и строки
        headers = (
            ["experiment", "epoch", "train_loss", "val_loss"]
            + [
                "val_iou_class_" + str(i)
                for i in range(best_metrics["val_metrics"]["IOU"].size(0))
            ]
            + [
                "val_f1_class_" + str(i)
                for i in range(best_metrics["val_metrics"]["F1"].size(0))
            ]
            + [
                "val_area_probs_f1_class_" + str(i)
                for i in range(best_metrics["val_metrics"]["area_probs_F1"].size(0))
            ]
            + ["val_mean_iou", "val_mean_f1", "val_mean_area_probs_f1"]
        )
        rows = []

    val_iou = [round(v.item(), 2) for v in best_metrics["val_metrics"]["IOU"]]
    val_f1 = [round(v.item(), 2) for v in best_metrics["val_metrics"]["F1"]]
    val_area_probs_f1 = [
        round(v.item(), 2) for v in best_metrics["val_metrics"]["area_probs_F1"]
    ]

    # Создаем строку с данными метрик
    row = (
        [
            best_metrics["experiment"],
            best_metrics["epoch"],
            round(best_metrics["train_loss"], 2),
            round(best_metrics["val_loss"], 2),
        ]
        # + best_metrics["val_metrics"]["IOU"].tolist()
        # + best_metrics["val_metrics"]["F1"].tolist()
        # + best_metrics["val_metrics"]["area_probs_F1"].tolist()
        + val_iou
        + val_f1
        + val_area_probs_f1
        + [
            round(best_metrics["val_metrics"]["IOU"].mean().item(), 2),
            round(best_metrics["val_metrics"]["F1"].mean().item(), 2),
            round(best_metrics["val_metrics"]["area_probs_F1"].mean().item(), 2),
        ]
    )

    # Проверяем, есть ли запись для текущего эксперимента, и обновляем ее
    experiment_exists = False
    for i, existing_row in enumerate(rows):
        if existing_row[0] == best_metrics["experiment"]:
            rows[i] = row
            experiment_exists = True
            break

    # Если записи для текущего эксперимента нет, добавляем новую строку
    if not experiment_exists:
        rows.append(row)

    # Записываем обновленные данные обратно в CSV файл
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


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


# def add_noise_to_mask(mask, noise_level=0.1):
#     mask = mask.float()
#     noise = torch.randn_like(mask) * noise_level
#     noisy_mask = mask + noise
#     return torch.clamp(noisy_mask, 0, 1)


def add_noise_to_mask(mask, max_m=3, max_n=3):
    mask = mask.float()  # Преобразуем маску к типу float
    noise = torch.rand_like(mask)  # Генерируем равномерный шум на [0, 1]
    m = torch.randint(1, max_m + 1, (1,)).item()  # Генерируем целое число от 1 до max_m включительно
    n = torch.randint(0, max_n + 1, (1,)).item()  # Генерируем целое число от 0 до max_n включительно
    noisy_mask = (m * mask + n * noise) / (m + n)
    return noisy_mask


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
):
    # Создание объекта SummaryWriter для записи логов
    writer = SummaryWriter(log_dir=f"runs_sinusite/{experiment_name}_logs")
    metrics_calculator = DetectionMetrics(mode="ML", num_classes=num_classes)

    class_names_dict = {
        class_info["id"]: class_info["name"] for class_info in sinusite_pat_classes_3
    }

    classes = list(class_names_dict.keys())
    weight_opt = Weight_opt_class(criterion, classes, None)

    # print("device", device)
    model = model.to(device)

    best_loss = 100

    if alpha is not None:
        alpha_no_fon = np.array([arr[1:] for arr in alpha])
        # alpha_no_fon = np.array(alpha[1:], dtype=np.float16) было так вместо верхней строки
        alpha_no_fon = torch.tensor(alpha_no_fon).to(device)
    else:
        alpha_no_fon = None

    if use_augmentation:
        seg_transform = SegTransform()

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
        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
        ) as pbar:
            for train_batch in train_loader:
                optimizer.zero_grad()
                images = train_batch["images"].to(device)
                masks = train_batch["masks"][:, 1:, :, :].to(device)
                # print("masks.shape", masks.shape)
                # print("images.shape", images.shape)

                if use_augmentation:
                    images, masks = seg_transform.apply_transform(images, masks)

                # шум к маске
                noisy_masks = add_noise_to_mask(masks, max_n, max_k)
                noisy_masks = noisy_masks.to(device)
                
                # шумная маска на изображение
                inputs = images + noisy_masks

                if all_class_weights is not None:
                    all_weights_no_fon = [x[1:] for x in all_class_weights]
                else:
                    all_weights_no_fon = None

                # print("alpha_no_fon shape", alpha_no_fon.shape)

                # print("masks shape", masks.shape) # torch.Size([16, 4, 256, 256])
                # print("images shape", images.shape) # torch.Size([64, 1, 256, 256])
                # print("images[0,0]", images[0,0])
                # values, counts = np.unique(images[0,0].cpu().numpy(), return_counts=True)
                # for v, c in zip(values, counts):
                #     print(f"v:{v}, c:{c}") # есть не нулей много
                # cv2.imwrite(f"image_wtf.jpg", images[0,0].cpu().numpy())
                # она просто черная вся

                # images = torch.cat((images, images, images, images), dim=1)  # Преобразование одноканального изображения в RGB
                # print("images shape", images.shape) # torch.Size([16, 3, 256, 256])

                # images = images.double()
                # model = model.double()
                outputs = model(inputs)
                # print("outputs shape", outputs.shape) # torch.Size([16, 5, 256, 256])
                # print("outputs before sigm", outputs)
                outputs = torch.tanh(outputs)
                # print("outputs after", outputs) # от 0 до 1


                # Вычитаем предсказанный шум из исходного изображения
                corrected_masks = (noisy_masks - outputs + 1) / 3

                # шумная маска
                # менять уровень шума радномно для каждой картинке 
                # надо чтобы сетка искала шум а потом вичитала его из маски 
                # в критерий подается начальная маска и noise_to_mask - шум
                loss = criterion(corrected_masks, masks, all_weights_no_fon, alpha_no_fon)

                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()

                train_iou_batch = iou_metric(corrected_masks, masks, num_classes)
                # train_iou_sum += torch.sum(train_iou_batch, dim=0)  # Суммирование IoU для каждого класса по всем батчам
                # вроде так
                train_iou_sum += train_iou_batch

                # для трейна метрики тоже посчитаю
                metrics_calculator.update_counter(
                    masks, corrected_masks
                )  # , advanced_metrics=True)

                # values, counts = np.unique(outputs.detach().cpu().numpy(), return_counts=True)
                # for v, c in zip(values, counts):
                #     print(f"v:{v}, c:{c}") # тут нули просто все
                # train_image_visualizer.visualize(images, masks, outputs, class_names_dict, colors, epoch)

                # pbar.set_postfix(loss=loss.item())

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
            train_metrics = metrics_calculator.calc_metrics()

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
                        writer.add_scalar(f"Train/{key}/Class_{i}", val.item(), epoch)
                else:
                    writer.add_scalar(f"Train/{key}", value.item(), epoch)

        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

        print("alpha_no_fon", alpha_no_fon)
        # оптимизация пиксельных весов
        if use_opt_pixel_weight:
            alpha_no_fon = weight_opt.opt_pixel_weight(train_metrics, alpha_no_fon)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss_avg}, Train IoU: {train_iou_avg}"
        )

        # Валидация
        model.eval()
        with torch.no_grad():
            # возможно это Саня добавил но по идее это не нужно у меня и так в начале каждой эпохи выше это написано
            # val_loss_sum = 0.0
            # val_iou_sum = 0.0 # добавил

            for k, val_batch in enumerate(val_loader):
                # for val_batch in val_loader:
                images_val = val_batch["images"].to(device)
                # rgb_image_val = val_batch["rgb_image"]
                # print("rgb_image_val", rgb_image_val)
                # print("images_val", images_val.shape) # torch.Size([32, 1, 256, 256])
                # print("images_val[0,0]", images_val[0,0].shape) # torch.Size([256, 256])
                # print("type mages_val", type(images_val))
                # values, counts = np.unique(images_val[0,0].cpu().numpy(), return_counts=True)
                # print("values", values) # тут не нули
                # print("images_val[0,0]", images_val[0,0]) # а тут нули
                masks_val = val_batch["masks"][:, 1:].to(device)


                # Добавляем шум один раз
                noisy_masks_val = add_noise_to_mask(masks_val, max_n, max_k)
                noisy_masks_val = noisy_masks_val.to(device)

                inputs_val = images_val + noisy_masks_val
                outputs_val = model(inputs_val)
                outputs_val = torch.tanh(outputs_val)
                corrected_masks_val = (noisy_masks_val - outputs_val + 1) / 3


                # Циклический процесс для предсказаний
                # сделать ноль этого цикла, на тесте сделаем цикл
                # for _ in range(num_cyclic_steps):
                #     outputs_val = model(images_val)
                #     outputs_val = torch.sigmoid(outputs_val)


                val_loss_sum += criterion(corrected_masks_val, masks_val, None, None).item()

                # val_iou_batch = iou_pytorch(outputs_val, masks_val, classes)

                val_iou_batch = iou_metric(corrected_masks_val, masks_val, num_classes)
                val_iou_sum += val_iou_batch

                metrics_calculator.update_counter(
                    masks_val, corrected_masks_val
                )  # advanced_metrics=True)

                # добавлю просто чтобы посмотреть
                # outputs_val_list.append(outputs_val)
                # masks_val_list.append(masks_val)

                # print("outputs_val", outputs_val)
                # values, counts = np.unique(outputs_val.cpu().numpy(), return_counts=True)
                # for v, c in zip(values, counts):
                #     print(f"v:{v}, c:{c}") # тут нули просто все

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
            val_metrics = metrics_calculator.calc_metrics()

        # было так
        # for key, value in val_metrics.items():
        #     if isinstance(value, torch.Tensor):
        #         writer.add_scalar(f"Val/{key}", value.mean().item(), epoch)
        #     elif isinstance(value, list):
        #         for i, val in enumerate(value):
        #             writer.add_scalar(f"Val/{key}/Class_{i}", val.item(), epoch)

        for key, value in val_metrics.items():
            if isinstance(value, torch.Tensor):
                if len(value.size()) > 0:
                    # добавил среднюю метрику по классам
                    writer.add_scalar(f"Val/Mean/{key}", value.mean().item(), epoch)
                    for i, val in enumerate(value):
                        writer.add_scalar(f"Val/{key}/Class_{i}", val.item(), epoch)
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
        # writer.add_scalar("My_classic_IoU/train", train_iou_avg, epoch)
        # writer.add_scalar("My_classic_IoU/validation", val_iou_avg, epoch) #

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

            best_model_path = "sinusite_best_models"
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

    last_model_path = "sinusite_last_models"
    if not os.path.exists(last_model_path):
        os.makedirs(last_model_path)

    torch.save(
        model.state_dict(), f"{last_model_path}/last_{experiment_name}_model.pth"
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

    # оптимизация старых метрик
    # def opt_pixel_weight(self, metrics, pixel_all_class_weights=None):
    #     recall = metrics["recall"]
    #     precession = metrics["precision"]  # раньше precession было
    #     f1_score = metrics["F1"]

    #     b = self.b

    #     if b is None:
    #         b = 1

    #     for image_class, cl_name in enumerate(self.classes):
    #         # print("image_class", image_class)
    #         neg_coef = 1
    #         pos_coef = 1

    #         if recall[image_class].item() != 0 and precession[image_class].item() != 0:
    #             print("recall и precision != 0")
    #             print("recall[image_class].item()", recall[image_class].item())
    #             print("precession[image_class].item()", precession[image_class].item())

    #             neg_coef = (
    #                 (1 / b) * recall[image_class].item() / f1_score[image_class].item()
    #             )
    #             pos_coef = (
    #                 (b) * precession[image_class].item() / f1_score[image_class].item()
    #             )
    #             print("neg_coef", neg_coef)
    #             print("pos_coef", pos_coef)

    #             xsd = recall[image_class].item() / precession[image_class].item()
    #             print("xsd", xsd)
    #             if xsd > 0.9 and xsd < 1.1:
    #                 neg_coef = 1
    #                 pos_coef = 1
    #             class_coef = pos_coef
    #             print("вот после изменений")
    #             print("neg_coef", neg_coef)
    #             print("pos_coef", pos_coef)
    #             print("class_coef", class_coef)

    #         else:
    #             print("recall или precision == 0")
    #             print("recall[image_class].item()", recall[image_class].item())
    #             print("precession[image_class].item()", precession[image_class].item())

    #             pos_coef = 2.0
    #             class_coef = 2.0
    #             neg_coef = 0.5
    #             print("neg_coef", neg_coef)
    #             print("pos_coef", pos_coef)
    #             print("class_coef", class_coef)

    #         if pixel_all_class_weights is not None:
    #             pixel_all_class_weights[0][image_class] *= pos_coef
    #             pixel_all_class_weights[1][image_class] *= neg_coef
    #             pixel_all_class_weights[2][image_class] *= class_coef

    #     return pixel_all_class_weights

    # оптимизация новых метрик

    def opt_pixel_weight(self, metrics, pixel_all_class_weights=None):
        recall = metrics["advanced_recall"]
        precession = metrics["advanced_precision"]  # раньше precession было
        F1Score = metrics["advanced_F1"]

        b = self.b

        if b is None:
            b = 1

        for image_class, cl_name in enumerate(self.classes):
            # print("image_class", image_class)
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

    batch_size = 6
    num_classes = 2

    params = {
        "json_file_path": "/home/imran-nasyrov/sinusite_json_data",
        "delete_list": [],
        "base_classes": sinusite_base_classes,
        "out_classes": sinusite_pat_classes_3,
        "dataloader": True,
        "resize": (1024, 1024),
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

    for train_batch in train_loader:
        images = train_batch["images"]
        masks = train_batch["masks"][:, 1:, :, :]
        # print("masks", masks.shape) torch.Size([6, 2, 1024, 1024])
        for i in range(len(images)):
            coco_dataloader.show_image_with_mask(
                images[i][0].cpu().numpy(), masks[i].cpu().numpy(), i
            )
        break  # Display only the first batch

    print("total_train", total_train)
    print("len total_train", len(total_train))
    print("list_of_name_out_classes", list_of_name_out_classes)
    print("pixel_TotalTrain", pixel_total_train)
    print("len val_loader", len(val_loader))
    print("len train_loader", len(train_loader))

    device = torch.device("cuda:1")
    print(device)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    model = smp.FPN(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=2,
        # classes=num_classes,
        classes=1,
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

    #     def forward(self, x):
    #         # Преобразование одноканального изображения в трехканальное
    #         x = x.repeat(1, 3, 1, 1)
    #         outputs = self.model(pixel_values=x)

    #         print("Type of outputs:", type(outputs))
    #         print("Keys in outputs:", outputs.keys())
    #         print("outputs.logits shape", outputs.logits.shape)
    #         return outputs.logits

    # model = FlorenceSegmentationModel(model).to(device)

    learning_rate = 3e-4
    num_epochs = 120

    optimizer = Adam(model.parameters(), lr=learning_rate)
    lr_sched = None

    use_class_weight = False
    use_pixel_weight = False
    use_pixel_opt = False
    power = "2.14_sinusite_weak"

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
    )

    # это картинки нарисует предсказанные

    model_weight = f"sinusite_best_models/best_{experiment_name}_model.pth"

    val_predict_path = f"predict_sinusite/predict_{experiment_name}/val"
    train_predict_path = f"predict_sinusite/predict_{experiment_name}/train"

    limited_train_loader = itertools.islice(train_loader, 6)
    limited_val_loader = itertools.islice(val_loader, 6)

    # avg_loss = test_model(
    #     model,
    #     model_weight,
    #     criterion,
    #     limited_train_loader,
    #     train_predict_path,
    #     limited_val_loader,
    #     val_predict_path,
    #     device,
    #     num_classes,
    # )
