import cv2
import random
import os
import numpy as np
import torchvision
import torch
import pickle
import hashlib


class Universal_npz_Segmentation_Dataset:
    def __init__(
        self,
        files_path,
        delete_list,
        base_classes,
        out_classes,
        dataloader=False,
        resize=None,
        recalculate=False,
    ):
        self.resize = resize
        self.files_path = files_path

        self.out_cl_hash = self.generate_class_id(out_classes)

        if self.files_path[-1] != r"/":
            self.files_path += r"/"

        self.images_paths = self.get_all_files(f"{self.files_path}images")
        self.masks_paths = self.get_all_files(f"{self.files_path}masks")
        self.all_images_and_mask_paths = None

        os.system(f"mkdir {self.files_path}hash")
        self.dataloader = dataloader

        self.delete_list = delete_list

        self.out_classes = out_classes

        self.base_classes = base_classes

        self.list_of_name_out_classes = ["фон"]
        self.list_of_name_base_classes = ["фон"]

        for segmentation_class in self.base_classes:
            self.list_of_name_base_classes.append(segmentation_class["name"])

        for segmentation_class in self.out_classes:
            self.list_of_name_out_classes.append(segmentation_class["name"])

        self.check_create_data(recalculate)
        self.check_create_train_val_list(recalculate)
        self.check_create_weight(recalculate)

    def __len__(self):
        return len(self.train_list) + len(self.val_list)

    def generate_class_id(self, classes):
        my_hash = 0
        for cl in classes:
            my_str = (
                str(cl["id"])
                + cl["name"]
                + cl["name"]
                + str(cl["summable_masks"])
                + str(cl["subtractive_masks"])
            )
            my_str = my_str.encode("utf-8")
            my_hash += int.from_bytes(
                bytes(hashlib.sha256(my_str).hexdigest(), "utf-8"), "little"
            )
        return str(my_hash)

    def get_all_files(self, directory):
        all_files = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)

        return all_files

    def check_create_train_val_list(self, recalculate):
        try:
            train_list = open(
                f"{self.files_path}hash/train_list_{self.out_cl_hash}.pickle", "rb"
            )
            val_list = open(
                f"{self.files_path}hash/val_list_{self.out_cl_hash}.pickle", "rb"
            )
            all_img_list = open(
                f"{self.files_path}hash/all_img_list_{self.out_cl_hash}.pickle", "rb"
            )

            self.train_list = pickle.load(train_list)
            self.val_list = pickle.load(val_list)
            self.all_img_list = pickle.load(all_img_list)
            print(self.out_cl_hash, recalculate)

            if recalculate != False:
                raise ValueError("stuff is not in content")

        except FileNotFoundError:
            num_imgs = len(self.all_images_and_mask_paths)
            print("Готовые данные не обраружены или включена рекалькуляция")
            train_list = []
            val_list = []
            all_img_list = []

            for img_index in range(num_imgs):
                all_img_list.append(img_index)
                x = random.randint(1, 100)
                if x >= 80:
                    val_list.append(img_index)
                else:
                    train_list.append(img_index)

            self.train_list = train_list
            self.val_list = val_list
            self.all_img_list = all_img_list

            with open(
                f"{self.files_path}hash/train_list_{self.out_cl_hash}.pickle", "wb"
            ) as train_list:
                pickle.dump(self.train_list, train_list)
            with open(
                f"{self.files_path}hash/val_list_{self.out_cl_hash}.pickle", "wb"
            ) as val_list:
                pickle.dump(self.val_list, val_list)
            with open(
                f"{self.files_path}hash/all_img_list_{self.out_cl_hash}.pickle", "wb"
            ) as all_img_list:
                pickle.dump(self.all_img_list, all_img_list)

    def check_create_weight(self, recalculate):
        try:
            TotalTrain = open(
                f"{self.files_path}hash/TotalTrain_{self.out_cl_hash}.pickle", "rb"
            )
            TotalVal = open(
                f"{self.files_path}hash/TotalVal_{self.out_cl_hash}.pickle", "rb"
            )
            pixel_TotalTrain = open(
                f"{self.files_path}hash/pixel_TotalTrain_{self.out_cl_hash}.pickle",
                "rb",
            )
            pixel_TotalVal = open(
                f"{self.files_path}hash/pixel_TotalVal_{self.out_cl_hash}.pickle", "rb"
            )

            self.TotalTrain = pickle.load(TotalTrain)
            self.TotalVal = pickle.load(TotalVal)
            self.pixel_TotalTrain = pickle.load(pixel_TotalTrain)
            self.pixel_TotalVal = pickle.load(pixel_TotalVal)
            print(recalculate)
            if recalculate != False:
                raise ValueError("stuff is not in content")

        except FileNotFoundError:
            print("готовые веса не обраружены или включена рекалькуляция")

            noc = len(self.list_of_name_out_classes)
            TotalTrain = np.zeros(noc)
            TotalVal = np.zeros(noc)

            pixel_TotalTrain = np.zeros(noc)
            pixel_TotalVal = np.zeros(noc)

            for i in self.train_list:
                image, mask = self.__getitem__(i, for_inner_use=True)
                for j in range(noc):
                    TotalTrain[j] += mask[j].max().item()
                    pixel_TotalTrain[j] += mask[j].sum().item()

            for i in self.val_list:
                image, mask = self.__getitem__(i, for_inner_use=True)
                for j in range(noc):
                    TotalVal[j] += mask[j].max().item()
                    pixel_TotalVal += mask[j].sum().item()

            self.TotalTrain = TotalTrain
            self.TotalVal = TotalVal
            self.pixel_TotalTrain = pixel_TotalTrain
            self.pixel_TotalVal = pixel_TotalVal

            with open(
                f"{self.files_path}hash/TotalTrain_{self.out_cl_hash}.pickle", "wb"
            ) as Total:
                pickle.dump(self.TotalTrain, Total)
            with open(
                f"{self.files_path}hash/TotalVal_{self.out_cl_hash}.pickle", "wb"
            ) as Total:
                pickle.dump(self.TotalVal, Total)
            with open(
                f"{self.files_path}hash/pixel_TotalTrain_{self.out_cl_hash}.pickle",
                "wb",
            ) as pix_Total:
                pickle.dump(self.pixel_TotalTrain, pix_Total)
            with open(
                f"{self.files_path}hash/pixel_TotalVal_{self.out_cl_hash}.pickle", "wb"
            ) as pix_Total:
                pickle.dump(self.pixel_TotalVal, pix_Total)

    def check_create_data(self, recalculate):
        try:
            all_images_and_mask_paths = open(
                f"{self.files_path}hash/all_images_and_mask_paths_{self.out_cl_hash}.pickle",
                "rb",
            )
            self.all_images_and_mask_paths = pickle.load(all_images_and_mask_paths)

            if recalculate != False:
                raise ValueError("stuff is not in content")
        except FileNotFoundError:
            "Данные не сгруппированы"
            all_images_and_mask_paths = self.configurated_data()
            self.all_images_and_mask_paths = all_images_and_mask_paths
            with open(
                f"{self.files_path}hash/all_images_and_mask_paths_{self.out_cl_hash}.pickle",
                "wb",
            ) as all_images_and_mask_paths:
                pickle.dump(self.all_images_and_mask_paths, all_images_and_mask_paths)

    def configurated_data(self):
        all_images_and_mask_paths = []
        masks_paths = self.masks_paths.copy()
        for x in self.images_paths:
            images_and_mask_paths = {}
            images_and_mask_paths["image"] = x
            images_and_mask_paths["mask"] = None
            new_masks_paths = []
            for y in masks_paths:
                imag_name = x.split("/")[-1]
                mask_name = y.split("/")[-1]
                imag_name = imag_name.split(".")[:-1]
                mask_name = mask_name.split(".")[:-1]
                imag_name = ".".join(imag_name)
                mask_name = ".".join(mask_name)

                if imag_name == mask_name:
                    images_and_mask_paths["mask"] = y
                else:
                    new_masks_paths.append(y)

            masks_paths = new_masks_paths
            all_images_and_mask_paths.append(images_and_mask_paths)

        return all_images_and_mask_paths

    def to_out_classes(self, mask):
        size = mask.size()
        new_mask = torch.zeros(len(self.out_classes) + 1, size[1], size[2])

        new_mask[0] = 1

        if mask.max() > 0:
            for out_class in self.out_classes:
                for i in out_class["summable_masks"]:
                    new_mask[out_class["id"], :, :][mask[i, :, :] == 1] = 1
                for i in out_class["subtractive_masks"]:
                    new_mask[out_class["id"], :, :][mask[i, :, :] == 1] = 0
                new_mask[0][new_mask[out_class["id"], :, :] == 1] = 0

        return new_mask

    def load_data(self, path_to_file):
        if path_to_file == None:
            data = np.zeros((len(self.base_classes), self.resize[0], self.resize[1]))
        else:
            if path_to_file.find("npz") != -1:
                data = np.load(path_to_file)
            elif path_to_file.find("jpg") != -1 or path_to_file.find("png") != -1:
                data = cv2.imread(path_to_file)
                data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        return data

    def __getitem__(self, idx, for_inner_use=False):
        image_path = self.all_images_and_mask_paths[idx]["image"]
        mask_path = self.all_images_and_mask_paths[idx]["mask"]

        image = self.load_data(image_path)
        image = torch.tensor(image)

        base_mask = self.load_data(mask_path)
        base_mask = torch.tensor(base_mask)

        image = torch.unsqueeze(image, 0)
        image = (image - image.min()) / (image.max() - image.min() + 0.00000001)

        if base_mask.dim() == 2:
            base_mask = torch.unsqueeze(base_mask, 0)

        mask = torch.zeros(
            len(self.base_classes) + 1, base_mask.size()[-2], base_mask.size()[-1]
        )
        mask[1:, :, :] = base_mask
        mask = self.to_out_classes(mask)

        if self.resize:
            image = torchvision.transforms.functional.resize(image, (self.resize))
            mask = torchvision.transforms.functional.resize(mask, (self.resize))

        if self.dataloader == False and for_inner_use == False:
            image = torch.unsqueeze(image, 0)
            mask = torch.unsqueeze(mask, 0)

        result = {}
        result["images"] = image
        result["masks"] = mask
        result["labels"] = torch.amax(mask, dim=(-1, -2))
        result["values"] = torch.sum(mask, (-1, -2))
        return result
