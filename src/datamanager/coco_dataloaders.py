import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .coco_classes import (
    SCT_base_classes,
    SCT_out_classes,
)
from .json_handler import JsonHandler


class KIDNEYS_COCODataLoader:
    def __init__(self, json_params, num_processes=1, rank=0):
        self.json_params = json_params
        self.num_processes = num_processes
        self.rank = rank
        self.list_out_classes = None

    @staticmethod
    def kidneys_get_target_directories(base_dir, num_series):
        target_dirs = []
        val_target_dirs = []

        # Получаем серию папок (P_1, P_2 и т.д.) и ограничиваем их числом num_series
        series_folders = sorted(
            [
                f for f in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, f)) and f.startswith("P_")
            ]
        )
        selected_series_folders = series_folders[:num_series]
        print("selected_series_folders", selected_series_folders)

        # Собираем целевые директории для выбранных серий папок
        for series_folder in selected_series_folders:
            series_path = os.path.join(base_dir, series_folder)
            for root, dirs, _ in os.walk(series_path):
                if any(view in root for view in ['axial', 'frontal', 'sagital']):
                    images_path = os.path.join(root, 'images')
                    annotations_path = os.path.join(root, 'annotations')
                    if os.path.isdir(images_path) and os.path.isdir(annotations_path):
                        target_dirs.append(root)

        val_dir = os.path.join(base_dir, 'val')
        if os.path.isdir(val_dir):
            for root, dirs, _ in os.walk(val_dir):
                if any(view in root for view in ['axial', 'frontal', 'sagital']):
                    images_path = os.path.join(root, 'images')
                    annotations_path = os.path.join(root, 'annotations')
                    if os.path.isdir(images_path) and os.path.isdir(annotations_path):
                        val_target_dirs.append(root)

        return target_dirs, val_target_dirs

    def class_instance(self, path, split_category):
        self.json_params["json_file_path"] = path

        sct_coco = JsonHandler(self.json_params, split_category)
        self.list_out_classes = sct_coco.list_out_classes
        return sct_coco

    def make_dataloaders(self, batch_size, num_series=5):
        target_dirs, val_target_dirs = self.kidneys_get_target_directories(self.json_params["json_file_path"], num_series)

        all_train_data = []
        all_val_data = []
        count = 0

        for subdir in target_dirs:
            try:
                print("Processing train folder:", subdir)
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
            except Exception as e:
                print(f"Error processing train folder {subdir}: {e}")

        # Обрабатываем валидационные папки
        for subdir in val_target_dirs:
            try:
                print("Processing val folder:", subdir)
                sct_coco = self.class_instance(subdir, "val")
                
                val_dataset = Subset(sct_coco, sct_coco.val_list)
                all_val_data.append(val_dataset)
                
                count += 1
            except Exception as e:
                print(f"Error processing val folder {subdir}: {e}")

        concat_train_data = ConcatDataset(all_train_data)
        concat_val_data = ConcatDataset(all_val_data)

        train_sampler = DistributedSampler(
            concat_train_data, num_replicas=self.num_processes, rank=self.rank
        )
        val_sampler = DistributedSampler(
            concat_val_data, num_replicas=self.num_processes, rank=self.rank
        )

        train_loader = DataLoader(
            concat_train_data,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=True,
            num_workers=4,
            collate_fn=self.custom_collate_fn,
        )
        val_loader = DataLoader(
            concat_val_data,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=4,
            collate_fn=self.custom_collate_fn,
        )

        return (
            train_loader,
            val_loader,
            total_train,
            pixel_total_train,
            self.list_out_classes,
        )
        
    @staticmethod
    def custom_collate_fn(batch):
        images = torch.stack([item["images"] for item in batch])
        masks = torch.stack([item["masks"] for item in batch])
        return {"images": images, "masks": masks}


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
        # это я сделал чтобы перенести косячные папки
        wrong_folder_path = "/home/imran-nasyrov/json_pocki_wrong_folders"
        # и понять какие папки создают ошибки, просто в почках некоторые данные плохие

        for subdir in train_folders:
            try:
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
            except Exception as e:
                print(f"Error processing folder {subdir}: {e}")
                # wrong_subdir_path = os.path.join(wrong_folder_path, os.path.basename(subdir))
                # shutil.move(subdir, wrong_subdir_path)

        for subdir in val_folders:
            try:
                print("subdir val", subdir)
                sct_coco = self.class_instance(subdir, "val")

                val_dataset = Subset(sct_coco, sct_coco.val_list)
                all_val_data.append(val_dataset)

                count += 1
            except Exception as e:
                print(f"Error processing folder {subdir}: {e}")
                # wrong_subdir_path = os.path.join(wrong_folder_path, os.path.basename(subdir))
                # shutil.move(subdir, wrong_subdir_path)

        concat_train_data = ConcatDataset(all_train_data)
        concat_val_data = ConcatDataset(all_val_data)

        train_loader = DataLoader(
            concat_train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self.custom_collate_fn,
        )
        val_loader = DataLoader(
            concat_val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self.custom_collate_fn,
        )

        return (
            train_loader,
            val_loader,
            total_train,
            pixel_total_train,
            self.list_out_classes,
        )

    @staticmethod
    def custom_collate_fn(batch):
        images = torch.stack([item["images"] for item in batch])
        masks = torch.stack([item["masks"] for item in batch])
        return {"images": images, "masks": masks}

    def show_image_with_mask(self, image, mask, idx):
        image = (image * 255).astype(np.uint8)
        mask = mask.astype(np.uint8)

        # Convert grayscale image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Define colors for the classes
        # Red for class 1, Blue for class 2
        colors = [(0, 0, 255), (255, 0, 0)]

        # Draw contours on the image
        for i in range(mask.shape[0]):
            contours, _ = cv2.findContours(
                mask[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image_rgb = cv2.drawContours(image_rgb, contours, -1, colors[i], 2)

        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        # Save the image with contours
        photo_path = "/home/imran-nasyrov/test_img_sin"
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
