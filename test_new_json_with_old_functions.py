import os
import numpy as np
import random
from torch.utils.data import DataLoader, ConcatDataset, Subset

from new_json_handler import JsonHandler
from utils import (
    SCT_base_classes,
    SCT_out_classes,
)


def get_direct_subdirectories(directory):
    subdirectories = [
        d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))
    ]
    return [os.path.join(directory, subdir) for subdir in subdirectories]


def convert_from_coco(path, split_category):
    params = {
        "json_file_path": path,
        "delete_list": [],
        "base_classes": SCT_base_classes,
        "out_classes": SCT_out_classes,
        "dataloader": True,
        "resize": (256, 256),
        "recalculate": True,
        "delete_null": False,
        # "train_val_probs": probs,
    }

    sct_coco = JsonHandler(params, split_category)

    return sct_coco


def make_dataloaders(subdirectories_list, batch_size):
    random.shuffle(subdirectories_list)

    num_train_folders = int(0.8 * len(subdirectories_list))
    train_folders = subdirectories_list[:num_train_folders]
    val_folders = subdirectories_list[num_train_folders:]

    print(val_folders, "val_folders")
    all_train_data = []
    all_val_data = []

    count = 0
    for s in train_folders:
        sub_subdirectories_list = get_direct_subdirectories(s)

        print("s", s)
        for i in sub_subdirectories_list:
            try:
                sct_coco = convert_from_coco(i, "train")

                if count == 0:
                    print("sct_coco._total_train", sct_coco.total_train)
                    TotalTrain = np.copy(sct_coco._total_train)
                    print(TotalTrain, "TotalTrain")
                    pixel_TotalTrain = np.copy(sct_coco._pixel_total_train)

                else:
                    print("sct_coco._total_train", sct_coco.total_train)
                    print(TotalTrain, "TotalTrain")
                    TotalTrain += sct_coco._total_train
                    pixel_TotalTrain += sct_coco._pixel_total_train

                train_dataset = Subset(sct_coco, sct_coco.train_list)
                all_train_data.append(train_dataset)

                count += 1
            except FileNotFoundError:
                print("no")

    for s in val_folders:
        print("s val", s)
        sub_subdirectories_list = get_direct_subdirectories(s)

        for i in sub_subdirectories_list:
            try:
                sct_coco = convert_from_coco(i, "val")

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


if __name__ == "__main__":
    path = "/home/imran/Документы/Innopolis/First_data_test/FINAL_CONVERT"
    subdirectories_list = get_direct_subdirectories(path)
    batch_size = 10

    (
        train_loader,
        val_loader,
        TotalTrain,
        pixel_TotalTrain,
        list_of_name_out_classes,
    ) = make_dataloaders(subdirectories_list, batch_size)

    print("pixel_TotalTrain", pixel_TotalTrain)
    print("list_of_name_out_classes", list_of_name_out_classes)
    print("len val_loader", len(val_loader))
    print("len train_loader", len(train_loader))
    print("TotalTrain", TotalTrain)
    print("len TotalTrain", len(TotalTrain))
