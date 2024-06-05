import random
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate
from torch.optim import Adam, AdamW
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import itertools

# from MedSAM.segment_anything import sam_model_registry


from metrics import Detection_metrics

from utils import iou_metric, ExperimentSetup


from test_new_json_handler import SINUSITE_COCODataLoader
from utils import set_seed

set_seed(64)

sinusite_base_classes = [
    {"name": "Правая гайморова пазуха (внешний контур)", "id": 1},
    {"name": "Левая гайморова пазуха (внешний контур)", "id": 2},
    {"name": "Левая лобная пазуха (внешний контур)", "id": 3},
    {"name": "Правая лобная пазуха (внешний контур)", "id": 4},
    {"name": "Правая гайморова пазуха (граница внутренней пустоты)", "id": 5},
    {"name": "Левая гайморова пазуха (граница внутренней пустоты)", "id": 6},
    {"name": "Левая лобная пазуха (граница внутренней пустоты)", "id": 7},
    {"name": "Правая лобная пазуха (граница внутренней пустоты)", "id": 8},
    {"name": "Снижение пневматизации околоносовых пазух", "id": 9},
    {"name": "Горизонтальный уровень жидкость-воздух", "id": 10},
    {"name": "Отсутствие пневматизации околоносовых пазух", "id": 11},
    {"name": "Иная патология", "id": 12},
    {"name": "Надпись", "id": 13},
]


sinusite_pat_classes_3 = [
    {
        "name": "Снижение пневматизации околоносовых пазух",
        "id": 1,
        "summable_masks": [9, 11],
        "subtractive_masks": [],
    },
    {
        "name": "Горизонтальный уровень жидкость-воздух",
        "id": 2,
        "summable_masks": [10],
        "subtractive_masks": [],
    },
]


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


def convert_from_coco(path, probs):
    # print("path", path)
    # print("number_papki", number_papki) # FINAL_CONVERT
    # print("second_chislo", second_chislo) # 100604476

    sct_coco = Universal_json_Segmentation_Dataset(
        json_file_path=path + "/",
        delete_list=[],
        base_classes=sinusite_base_classes,
        out_classes=sinusite_pat_classes_3,
        delete_null=False,  # Fasle всегда
        resize=(1024, 1024),
        dataloader=True,
        recalculate=True,  # оставить True
        train_val_probs=probs,
    )

    first_item = sct_coco[1]

    image, target = first_item["images"], first_item["masks"]
    print(f"Image size: {image.shape}, Target size: {target.shape}")

    return sct_coco


def make_dataloaders(subdirectories_list, batch_size):
    random.shuffle(subdirectories_list)

    num_train_folders = int(0.9 * len(subdirectories_list))  # тут сделал 90% на вал
    train_folders = subdirectories_list[:num_train_folders]
    val_folders = subdirectories_list[num_train_folders:]

    all_train_data = []
    all_val_data = []

    count = 0
    # for s in train_folders:
    #     sub_subdirectories_list = get_direct_subdirectories(s)
    # print("sub_subdirectories_list", sub_subdirectories_list)
    print("train_folders", train_folders)
    print("val_folders", val_folders)
    print("len train_folders", len(train_folders))
    print("len val_folders", len(val_folders))
    # sub_subdirectories_list = get_direct_subdirectories(train_folders)

    for i in train_folders:
        print("i train", i)
        # try:
        sct_coco = convert_from_coco(i, 100)

        if count == 0:
            TotalTrain = np.copy(sct_coco.TotalTrain)
            pixel_TotalTrain = np.copy(sct_coco.pixel_TotalTrain)
        else:
            TotalTrain += sct_coco.TotalTrain
            pixel_TotalTrain += sct_coco.pixel_TotalTrain

        train_dataset = Subset(sct_coco, sct_coco.train_list)
        # for i in range(1, len(train_dataset)):
        #     sct_coco.show_me_contours(i)
        # print(train_dataset[i])
        # first_item = sct_coco[i]
        # image, target = first_item['images'], first_item['masks']
        # # gray_image, mask, rgb_image = sct_coco[i]
        # print(f"Image size: {image.shape}, Target size: {target.shape}")
        # # print("gray_image", gray_image.shape, "mask", mask.shape, "rgb_image", rgb_image.shape)

        all_train_data.append(train_dataset)

        count += 1
        # except:
        #     print("no")

    # for s in val_folders:
    #     sub_subdirectories_list = get_direct_subdirectories(s)
    # sub_subdirectories_list = get_direct_subdirectories(val_folders)
    for i in val_folders:
        print("i val", i)
        # try:
        sct_coco = convert_from_coco(i, 0)

        val_dataset = Subset(sct_coco, sct_coco.val_list)
        all_val_data.append(val_dataset)

        count += 1
        # except:
        #     print("no")

    concat_train_data = ConcatDataset(all_train_data)
    concat_val_data = ConcatDataset(all_val_data)

    train_loader = DataLoader(
        concat_train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        concat_val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn,
    )

    return (
        train_loader,
        val_loader,
        TotalTrain,
        pixel_TotalTrain,
        sct_coco.list_of_name_out_classes,
    )


def return_dataset_list_and_num_of_classes(
    sinusite_datasets_path, resize, recalculate, batch_size, mix_test, device=None
):
    DirPaths = os.listdir(sinusite_datasets_path)

    val_list_of_dataset = []
    train_list_of_dataset = []
    test_list_of_dataset = []
    for count, path in enumerate(DirPaths):
        print(path)
        if path.find("test") == -1 or mix_test == True:
            dataset = Universal_json_Segmentation_Dataset(
                json_file_path=sinusite_datasets_path + path,
                delete_list=[],
                base_classes=sinusite_base_classes,
                out_classes=sinusite_pat_classes_3,
                dataloader=True,
                resize=resize,
                recalculate=recalculate,
            )

            val_dataset = torch.utils.data.Subset(dataset, dataset.val_list)
            val_list_of_dataset.append(val_dataset)
            Total_val = np.copy(dataset.TotalVal)

            train_dataset = torch.utils.data.Subset(dataset, dataset.train_list)

            # for i in range(1, len(train_dataset)):
            #         first_item = dataset[i]
            #         image, target = first_item['images'], first_item['masks']
            #         print(f"Image size: {image.shape}, Target size: {target.shape}")

            train_list_of_dataset.append(train_dataset)

            if path.find("test") != -1 and mix_test == True:
                train_list_of_dataset.append(val_dataset)

            if count == 0:
                TotalTrain = np.copy(dataset.TotalTrain)
                pixel_TotalTrain = np.copy(dataset.pixel_TotalTrain)
            else:
                TotalTrain += dataset.TotalTrain
                pixel_TotalTrain += dataset.pixel_TotalTrain

    train_dataset = torch.utils.data.ConcatDataset(train_list_of_dataset)
    val_dataset = torch.utils.data.ConcatDataset(val_list_of_dataset)

    # Дополнительные проверки
    # for i in tqdm(range(1, len(train_dataset))):
    #     item = train_dataset[i]
    #     image, target = item['images'].to(device), item['masks'].to(device)
    #     if image.shape != torch.Size([1, 512, 512]) or target.shape != torch.Size([3, 512, 512]):
    #         print(f"Mismatch in concatenated train dataset at index {i}: Image size {image.shape}, Target size {target.shape}")

    # for i in tqdm(range(1, len(val_dataset))):
    #     item = val_dataset[i]
    #     image, target = item['images'].to(device), item['masks'].to(device)
    #     if image.shape != torch.Size([1, 512, 512]) or target.shape != torch.Size([3, 512, 512]):
    #         print(f"Mismatch in concatenated val dataset at index {i}: Image size {image.shape}, Target size {target.shape}")

    train_DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )
    val_DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )

    if len(test_list_of_dataset) != 0:
        test_dataset = torch.utils.data.ConcatDataset(test_list_of_dataset)
        test_DataLoader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=custom_collate_fn,
        )
    else:
        test_DataLoader = None

    return (
        train_DataLoader,
        val_DataLoader,
        test_list_of_dataset,
        TotalTrain,
        pixel_TotalTrain,
        dataset.list_of_name_out_classes,
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
    # Создание объекта SummaryWriter для записи логов
    writer = SummaryWriter(log_dir=f"runs_sinusite/{experiment_name}_logs")
    metrics_calculator = Detection_metrics(mode="ML", num_classes=num_classes)

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
                # rgb_image = train_batch["rgb_image"]
                # print("rgb_image", rgb_image.shape) # torch.Size([32, 512, 512, 3])
                # print("rgb_image", rgb_image[0, :, : ,0].shape) # torch.Size([512, 512])
                # cv2.imwrite(f"/home/imran-nasyrov/sct_project/sct_data/train_rgb_images/rgb_image[0,0].jpg", rgb_image[0].cpu().numpy())
                # # это нормально сохраняет

                # masks = masks[:][1:] # убрал фон
                # masks = masks.to(device)
                masks = train_batch["masks"][:, 1:, :, :].to(device)
                # print("masks.shape", masks.shape)
                # print("images.shape", images.shape)

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

            best_model_path = "sinusite_best_models"
            if not os.path.exists(best_model_path):
                os.makedirs(best_model_path)

            torch.save(
                model.state_dict(),
                f"{best_model_path}/best_{experiment_name}_model.pth",
            )

    last_model_path = "sinusite_last_models"
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

        if b == None:
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
        "recalculate": True,
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

    print("total_train", total_train)
    print("len total_train", len(total_train))
    print("list_of_name_out_classes", list_of_name_out_classes)
    print("pixel_TotalTrain", pixel_total_train)
    print("len val_loader", len(val_loader))
    print("len train_loader", len(train_loader))

    # было так но я класс сделал
    # (
    #     train_loader,
    #     val_loader,
    #     TotalTrain,
    #     pixel_TotalTrain,
    #     list_of_name_out_classes,
    # ) = make_dataloaders(subdirectories_list, batch_size)
    #
    # print("TotalTrain", TotalTrain)
    # print("len TotalTrain", len(TotalTrain))
    # print("len(train_loader)", len(train_loader))
    # print("len(val_loader)", len(val_loader))

    device = torch.device("cuda:0")
    print(device)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    # for i, (images, masks) in enumerate(train_loader):
    #     images, masks = images.to(device), masks.to(device)
    #     if images.shape != torch.Size([1, 512, 512]) or masks.shape != torch.Size([3, 512, 512]):
    #         print(f"Mismatch in concatenated train dataset at index {i}: Image size {masks.shape}, Target size {masks.shape}")

    model = smp.FPN(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=1,
        classes=num_classes,
    )
    # model = smp.Unet(encoder_name="resnext50_32x4d", encoder_weights = "imagenet", in_channels = 1, classes=num_classes)
    learning_rate = 3e-4
    num_epochs = 120

    optimizer = Adam(model.parameters(), lr=learning_rate)
    lr_sched = None

    use_class_weight = True
    use_pixel_weight = True
    use_pixel_opt = True
    power = "1.1_new_code_sinusite_weak"

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
