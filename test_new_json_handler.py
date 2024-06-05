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
        self.list_out_classes = None
        self.subdirectories = self.get_subdirectories(
            self.json_params["json_file_path"]
        )

    def get_subdirectories(self, directory):
        subdirectories = [
            d
            for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))
        ]
        return [os.path.join(directory, subdir) for subdir in subdirectories]

    def class_instance(self, path, split_category):
        self.json_params["json_file_path"] = path

        sct_coco = JsonHandler(self.json_params, split_category)
        self.list_out_classes = sct_coco.list_out_classes
        return sct_coco

    def make_dataloaders(self, batch_size, train_val_ratio=0.8):
        random.shuffle(self.subdirectories)

        num_folders = int(train_val_ratio * len(self.subdirectories))
        train_folders = self.subdirectories[:num_folders]
        val_folders = self.subdirectories[num_folders:]

        all_train_data = []
        all_val_data = []

        count = 0
        for s in train_folders:
            sub_subdirectories = self.get_subdirectories(s)

            print("s", s)
            for i in sub_subdirectories:
                try:
                    sct_coco = self.class_instance(i, "train")

                    if count == 0:
                        total_train = np.copy(sct_coco.total_train)
                        pixel_total_train = np.copy(sct_coco.pixel_total_train)
                    else:
                        print("sct_coco._total_train", sct_coco.total_train)
                        total_train += sct_coco.total_train
                        pixel_total_train += sct_coco.pixel_total_train

                    train_dataset = Subset(sct_coco, sct_coco.train_list)
                    all_train_data.append(train_dataset)

                    count += 1
                except FileNotFoundError:
                    print("no")

        for s in val_folders:
            print("s val", s)
            sub_subdirectories = self.get_subdirectories(s)

            for i in sub_subdirectories:
                try:
                    sct_coco = self.class_instance(i, "val")

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
            total_train,
            pixel_total_train,
            self.list_out_classes,
        )


if __name__ == "__main__":
    print("Hello")
    params = {
        "json_file_path": "/home/imran/Документы/Innopolis/First_data_test/FINAL_CONVERT",
        "delete_list": [],
        "base_classes": SCT_base_classes,
        "out_classes": SCT_out_classes,
        "dataloader": True,
        "resize": (256, 256),
        "recalculate": True,
        "delete_null": False,
    }

    coco_dataloader = COCODataLoader(params)

    (
        train_loader,
        val_loader,
        total_train,
        pixel_total_train,
        list_of_name_out_classes,
    ) = coco_dataloader.make_dataloaders(2, 0.8)

    print("total_train", total_train)
    print("len total_train", len(total_train))
    print("list_of_name_out_classes", list_of_name_out_classes)
    print("pixel_TotalTrain", pixel_total_train)
    print("len val_loader", len(val_loader))
    print("len train_loader", len(train_loader))
