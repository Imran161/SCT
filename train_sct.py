import itertools
import os
import random
import warnings

import numpy as np
import segmentation_models_pytorch as smp
import torch
from sklearn.exceptions import UndefinedMetricWarning
from torch.optim import Adam, AdamW
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from json_handler import JsonHandler
from metrics import DetectionMetrics
from sct_val import test_model
from utils import (
    ExperimentSetup,
    ImageVisualizer,
    SCT_base_classes,
    SCT_out_classes,
    iou_metric,
    set_seed,
)


def get_direct_subdirectories(directory):
    subdirectories = [
        d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))
    ]
    return [os.path.join(directory, subdir) for subdir in subdirectories]


def convert_from_coco(path, probs):
    # print("path", path)
    # print("number_papki", number_papki) # FINAL_CONVERT
    # print("second_chislo", second_chislo) # 100604476

    # вот так было для старого класса
    sct_coco = JsonHandler(
        json_file_path=path + "/",
        delete_list=[],
        base_classes=SCT_base_classes,
        out_classes=SCT_out_classes,
        delete_null=False,  # Fasle всегда
        resize=(256, 256),
        dataloader=True,
        recalculate=False,  # оставить True
        train_val_probs=probs,
    )

    # params = {
    #     "json_file_path": path,
    #     "delete_list": [],
    #     "base_classes": SCT_base_classes,
    #     "out_classes": SCT_out_classes,
    #     "dataloader": False,
    #     "resize": (256, 256),
    #     "recalculate": False,
    #     "delete_null": False,
    #     "train_val_probs": probs,
    # }
    #
    # sct_coco = JsonHandler(params)

    return sct_coco


def make_dataloaders(subdirectories_list, batch_size):
    random.shuffle(subdirectories_list)

    num_train_folders = int(0.8 * len(subdirectories_list))
    train_folders = subdirectories_list[:num_train_folders]
    val_folders = subdirectories_list[num_train_folders:]

    all_train_data = []
    all_val_data = []

    count = 0
    for s in train_folders:
        sub_subdirectories_list = get_direct_subdirectories(s)

        print("s", s)
        for i in sub_subdirectories_list:
            try:
                sct_coco = convert_from_coco(i, 100)

                if count == 0:
                    TotalTrain = np.copy(sct_coco.TotalTrain)
                    pixel_TotalTrain = np.copy(sct_coco.pixel_TotalTrain)
                else:
                    TotalTrain += sct_coco.TotalTrain
                    pixel_TotalTrain += sct_coco.pixel_TotalTrain

                train_dataset = Subset(sct_coco, sct_coco.train_list)
                all_train_data.append(train_dataset)

                count += 1
            except FileNotFoundError:
                print("no")

    for s in val_folders:
        sub_subdirectories_list = get_direct_subdirectories(s)

        for i in sub_subdirectories_list:
            try:
                sct_coco = convert_from_coco(i, 0)

                val_dataset = Subset(sct_coco, sct_coco.val_list)
                all_val_data.append(val_dataset)

                count += 1
            except FileNotFoundError:
                print("no")

    concat_train_data = ConcatDataset(all_train_data)
    concat_val_data = ConcatDataset(all_val_data)

    train_loader = DataLoader(
        concat_train_data, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        concat_val_data, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return (
        train_loader,
        val_loader,
        TotalTrain,
        pixel_TotalTrain,
        sct_coco.list_of_name_out_classes,
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
):
    train_predict_path = f"/home/imran-nasyrov/sct_project/sct_data/predict_test/train"
    train_image_visualizer = ImageVisualizer(train_predict_path)

    val_predict_path = f"/home/imran-nasyrov/sct_project/sct_data/predict_test/val"
    val_image_visualizer = ImageVisualizer(val_predict_path)

    # all_weights_no_fon = []

    # Создание объекта SummaryWriter для записи логов
    writer = SummaryWriter(log_dir=f"runs/{experiment_name}_logs")
    metrics_calculator = DetectionMetrics(mode="ML", num_classes=num_classes)

    class_names_dict = {
        class_info["id"]: class_info["name"] for class_info in SCT_out_classes
    }
    # вот тут на самом деле insult_type_1 стоят и тд я ниже написал просто как мне надо было
    # print("class_names_dict", class_names_dict) # {1: 'Внутримозговое кровозлияние', 2: 'Субарахноидальное кровозлияние', 3: 'Cубдуральное кровозлияние', 4: 'Эпидуральное кровозлияние'}

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
                # images = images.to(device)
                rgb_image = train_batch["rgb_image"]
                # print("rgb_image", rgb_image.shape) # torch.Size([32, 512, 512, 3])
                # print("rgb_image", rgb_image[0, :, : ,0].shape) # torch.Size([512, 512])
                # cv2.imwrite(f"/home/imran-nasyrov/sct_project/sct_data/train_rgb_images/rgb_image[0,0].jpg", rgb_image[0].cpu().numpy())
                # # это нормально сохраняет

                # masks = masks[:][1:] # убрал фон
                # masks = masks.to(device)
                masks = train_batch["masks"][:, 1:, :, :].to(device)

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
                outputs = model(images)
                # print("outputs shape", outputs.shape) # torch.Size([16, 5, 256, 256])
                # print("outputs before sigm", outputs)
                outputs = torch.sigmoid(outputs)
                # print("outputs after", outputs) # от 0 до 1

                loss = criterion(outputs, masks, all_weights_no_fon, alpha_no_fon)

                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()

                train_iou_batch = iou_metric(outputs, masks, num_classes)
                # train_iou_sum += torch.sum(train_iou_batch, dim=0)  # Суммирование IoU для каждого класса по всем батчам
                # вроде так
                train_iou_sum += train_iou_batch

                # для трейна метрики тоже посчитаю
                metrics_calculator.update_counter(masks, outputs)

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
            train_metrics = metrics_calculator.calculate_metrics()

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
                outputs_val = model(images_val)

                outputs_val = torch.sigmoid(outputs_val)

                val_loss_sum += criterion(outputs_val, masks_val, None, None).item()

                # val_iou_batch = iou_pytorch(outputs_val, masks_val, classes)

                val_iou_batch = iou_metric(outputs_val, masks_val, num_classes)
                val_iou_sum += val_iou_batch

                metrics_calculator.update_counter(masks_val, outputs_val)

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
            metrics = metrics_calculator.calculate_metrics()

        # было так
        # for key, value in metrics.items():
        #     if isinstance(value, torch.Tensor):
        #         writer.add_scalar(f"Val/{key}", value.mean().item(), epoch)
        #     elif isinstance(value, list):
        #         for i, val in enumerate(value):
        #             writer.add_scalar(f"Val/{key}/Class_{i}", val.item(), epoch)

        for key, value in metrics.items():
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

            best_model_path = "best_models"
            if not os.path.exists(best_model_path):
                os.makedirs(best_model_path)

            torch.save(
                model.state_dict(),
                f"{best_model_path}/best_{experiment_name}_model.pth",
            )

    last_model_path = "last_models"
    if not os.path.exists(last_model_path):
        os.makedirs(last_model_path)

    torch.save(
        model.state_dict(), f"{last_model_path}/last_{experiment_name}_model.pth"
    )

    writer.close()


class Weight_opt_class:
    def __init__(self, loss, classes, b=None):
        self.b = b
        self.loss = loss
        self.loss_class = loss
        self.classes = classes

    # def mask2label(self, mask):
    #     label = torch.amax(mask, [-1,-2])
    #     return label

    def opt_pixel_weight(self, metrics, pixel_all_class_weights=None):
        recall = metrics["recall"]
        precession = metrics["precession"]
        F1Score = metrics["F1"]

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
    set_seed(64)
    path = "/home/imran-nasyrov/sct_project/sct_data/FINAL_CONVERT"
    subdirectories_list = get_direct_subdirectories(path)
    batch_size = 100
    num_classes = 4
    # было так но я класс сделал
    (
        train_loader,
        val_loader,
        TotalTrain,
        pixel_TotalTrain,
        list_of_name_out_classes,
    ) = make_dataloaders(subdirectories_list, batch_size)
    # print("TotalTrain", TotalTrain) # [131669.  19075.   4210.   4014.   2525.]
    # print("len TotalTrain", len(TotalTrain)) # 5
    # так что теперь так

    # train_loader_iter = iter(train_loader)
    # batch = next(train_loader_iter)
    # print(len(batch["rgb_image"])) # 64 # images было
    # print(len(batch["masks"])) # 64

    device = torch.device("cuda:2")
    print(device)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    # потом можно попробовать трансформер обучить
    model = smp.FPN(
        encoder_name="mit_b5",
        encoder_depth=5,
        encoder_weights="imagenet",
        in_channels=1,
        classes=num_classes,
    )
    # encoder_depth = 5 у них написать что с FPN только с таким параметром работает

    model = smp.FPN(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=1,
        classes=num_classes,
    )
    learning_rate = 3e-4
    num_epochs = 120

    optimizer = Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)

    # я тут поменял местами значения base_lr и max_lr, чтобы lr сначала падал а потом рос
    # lr_sched = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-2, max_lr=learning_rate, step_size_up=15, step_size_down=15,
    #                                              mode='triangular2', cycle_momentum=False)
    # можно такой
    # lr_sched = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=1.585)
    lr_sched = None

    use_class_weight = True
    use_pixel_weight = True
    use_pixel_opt = True
    power = "strong"  # 3.1 забыл написать

    exp_setup = ExperimentSetup(
        train_loader, TotalTrain, pixel_TotalTrain, batch_size, num_classes
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
    )

    # это картинки нарисует предсказанные

    model_weight = f"best_{experiment_name}_model.pth"

    val_predict_path = f"predict/predict_{experiment_name}/val"
    train_predict_path = f"predict/predict_{experiment_name}/train"

    limited_train_loader = itertools.islice(train_loader, 16)
    limited_val_loader = itertools.islice(val_loader, 16)

    # avg_loss = test_model(model, model_weight, criterion,
    #                         limited_train_loader, train_predict_path,
    #                         limited_val_loader, val_predict_path, device, num_classes)
