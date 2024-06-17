import os
import numpy as np
import random
from torch.utils.data import DataLoader, ConcatDataset, Subset
import cv2
import matplotlib.pyplot as plt

from new_json_handler import JsonHandler
# from json_handler import JsonHandler
from utils import (
    SCT_base_classes,
    SCT_out_classes,
)


class SINUSITE_old_COCODataLoader:
    def __init__(self, json_params):
        self.json_params = json_params
        self.list_out_classes = None
        self.subdirectories = self.get_subdirs(self.json_params["json_file_path"])

    def get_subdirs(self, directory):
        subdirectories = [
            d
            for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))
        ]
        return [os.path.join(directory, subdir) for subdir in subdirectories]

    def class_instance(self, path, split_category):
        self.json_params["json_file_path"] = path

        sct_coco = JsonHandler(self.json_params)#, split_category)
        self.list_out_classes = sct_coco.list_of_name_out_classes # sct_coco.list_out_classes
        return sct_coco

    def make_dataloaders(self, batch_size, train_val_ratio=0.8):
        random.shuffle(self.subdirectories)

        num_folders = int(train_val_ratio * len(self.subdirectories))
        train_folders = self.subdirectories[:num_folders]
        val_folders = self.subdirectories[num_folders:]

        all_train_data = []
        all_val_data = []

        count = 0
        for subdir in train_folders:
            print("subdir train", subdir)
            sct_coco = self.class_instance(subdir, "train")
            
            print("self.json_params[json_file_path]", self.json_params["json_file_path"])

            if count == 0:
                total_train = np.copy(sct_coco.TotalTrain)
                pixel_total_train = np.copy(sct_coco.pixel_TotalTrain)
            else:
                print("sct_coco._total_train", sct_coco.TotalTrain)
                total_train += sct_coco.TotalTrain
                pixel_total_train += sct_coco.pixel_TotalTrain

            train_dataset = Subset(sct_coco, sct_coco.TotalTrain)
            all_train_data.append(train_dataset)

            count += 1

        for subdir in val_folders:
            print("subdir val", subdir)
            sct_coco = self.class_instance(subdir, "val")

            val_dataset = Subset(sct_coco, sct_coco.val_list)
            all_val_data.append(val_dataset)

            count += 1

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

# в новом моем json неправильные маски
class SINUSITE_COCODataLoader:
    def __init__(self, json_params):
        self.json_params = json_params
        self.list_out_classes = None
        self.subdirectories = self.get_subdirs(self.json_params["json_file_path"])

    def get_subdirs(self, directory):
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
        for subdir in train_folders:
            print("subdir train", subdir)
            sct_coco = self.class_instance(subdir, "train")

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

        for subdir in val_folders:
            print("subdir val", subdir)
            sct_coco = self.class_instance(subdir, "val")

            val_dataset = Subset(sct_coco, sct_coco.val_list)
            all_val_data.append(val_dataset)

            count += 1

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
        
    def show_image_with_mask(self, image, mask, idx):
        image = (image * 255).astype(np.uint8)
        mask = mask.astype(np.uint8)
        
        # Convert grayscale image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Define colors for the classes
        colors = [(0, 0, 255), (255, 0, 0)]  # Red for class 1, Blue for class 2
        
        # Draw contours on the image
        for i in range(mask.shape[0]):
            contours, _ = cv2.findContours(mask[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            image_rgb = cv2.drawContours(image_rgb, contours, -1, colors[i], 2)
        
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        
        # Save the image with contours
        photo_path = f"/home/imran-nasyrov/test_img_sin"
        if not os.path.exists(photo_path):
            os.makedirs(photo_path)
                
        cv2.imwrite(f"{photo_path}/test_img_sin_{idx}.jpg", image_rgb)


class SCT_COCODataLoader:
    def __init__(self, json_params):
        self.json_params = json_params
        self.list_out_classes = None
        self.subdirectories = self.get_subdirs(self.json_params["json_file_path"])

    def get_subdirs(self, directory):
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
        for subdir in train_folders:
            sub_subdirs = self.get_subdirs(subdir)

            print("subdir train", subdir)
            for sub_subdir in sub_subdirs:
                path = os.path.join(sub_subdir, "annotations/instances_default.json")

                if os.path.exists(path):
                    sct_coco = self.class_instance(sub_subdir, "train")

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
                else:
                    print(f"File not found for directory: {sub_subdir}")

        for subdir in val_folders:
            print("subdir val", subdir)
            sub_subdirs = self.get_subdirs(subdir)

            for sub_subdir in sub_subdirs:
                path = os.path.join(sub_subdir, "annotations/instances_default.json")

                if os.path.exists(path):
                    sct_coco = self.class_instance(sub_subdir, "val")

                    val_dataset = Subset(sct_coco, sct_coco.val_list)
                    all_val_data.append(val_dataset)

                    count += 1
                else:
                    print(f"File not found for directory: {sub_subdir}")

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

    coco_dataloader = SCT_COCODataLoader(params)

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
