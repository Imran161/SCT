import os
import numpy as np
import random
from torch.utils.data import DataLoader, ConcatDataset, Subset

from new_json_handler import JsonHandler
from utils import (
    SCT_base_classes,
    SCT_out_classes,
)


class COCODataLoader:
    def __init__(self, json_params):
        self.json_params = json_params
        self.subdirectories_list = self.get_direct_subdirectories(
            self.json_params["json_file_path"]
        )
        # self.handler = JsonHandler(json_params)

    def get_direct_subdirectories(self, directory):
        subdirectories = [
            d
            for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))
        ]
        return [os.path.join(directory, subdir) for subdir in subdirectories]

    def convert_from_coco(self, path, probs):
        self.json_params["train_val_probs"] = probs
        self.json_params["json_file_path"] = path

        sct_coco = JsonHandler(self.json_params)
        return sct_coco

    def make_dataloaders(self, batch_size, train_val_ratio=0.8):
        random.shuffle(self.subdirectories_list)

        num_train_folders = int(train_val_ratio * len(self.subdirectories_list))
        train_folders = self.subdirectories_list[:num_train_folders]
        val_folders = self.subdirectories_list[num_train_folders:]

        all_train_data = []
        all_val_data = []

        count = 0
        for s in train_folders:
            sub_subdirectories_list = self.get_direct_subdirectories(s)

            print("s", s)
            for i in sub_subdirectories_list:
                try:
                    sct_coco = self.convert_from_coco(i, 100)

                    if count == 0:
                        TotalTrain = np.copy(sct_coco._total_train)
                        pixel_TotalTrain = np.copy(sct_coco._pixel_total_train)
                    else:
                        print("sct_coco._total_train", sct_coco.total_train)
                        TotalTrain += sct_coco._total_train
                        pixel_TotalTrain += sct_coco._pixel_total_train

                    train_dataset = Subset(sct_coco, sct_coco.train_list)
                    all_train_data.append(train_dataset)

                    count += 1
                except FileNotFoundError:
                    print("no")

        for s in val_folders:
            sub_subdirectories_list = self.get_direct_subdirectories(s)

            for i in sub_subdirectories_list:
                try:
                    sct_coco = self.convert_from_coco(i, 0)

                    val_dataset = Subset(sct_coco, sct_coco._val_list)
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
            self.handler.list_of_name_out_classes,
        )


if __name__ == "__main__":
    params = {
        "json_file_path": "/home/imran/Документы/Innopolis/First_data_test/FINAL_CONVERT",
        "delete_list": [],
        "base_classes": SCT_base_classes,
        "out_classes": SCT_out_classes,
        "dataloader": True,
        "resize": (256, 256),
        "recalculate": False,
        "delete_null": False,
        # "train_val_probs": 80, он тут не нужен,
        # тк в make_dataloaders передается параметр для разделения
    }

    coco_dataloader = COCODataLoader(params)

    (
        train_loader,
        val_loader,
        TotalTrain,
        pixel_TotalTrain,
        list_of_name_out_classes,
    ) = coco_dataloader.make_dataloaders(10, 0.8)

    print("TotalTrain", TotalTrain)
    print("len TotalTrain", len(TotalTrain))
    print("list_of_name_out_classes", list_of_name_out_classes)
    # handler = JsonHandler(params)
    #
    # # Получение списков изображений для обучения и валидации через свойства
    # train_list = handler.train_list
    # val_list = handler.val_list
    #
    # # Работа с первым изображением из обучающего списка
    # img_id = train_list[0]
    #
    # # Получение изображения и масок
    # gray_image, mask, rgb_image = handler[img_id]
    #
    # # Преобразование маски в выходные классы
    # new_mask = handler.to_out_classes(mask)
    #
    # # Получение веса классов через свойства
    # total_train = handler.total_train
    # total_val = handler.total_val
    # pixel_total_train = handler.pixel_total_train
    # pixel_total_val = handler.pixel_total_val
    #
    # # Вывод результатов
    # print(f"Общее количество тренировочных изображений: {len(train_list)}")
    # print(f"Общее количество валидационных изображений: {len(val_list)}")
    # print(f"Размеры серого изображения: {gray_image.shape}")
    # print(f"Размеры маски: {mask.shape}")
    # print(f"Новые маски для выходных классов: {new_mask.shape}")
    # print(f"Общий вес тренировочных данных: {total_train}")
    # print(f"Общий вес валидационных данных: {total_val}")
