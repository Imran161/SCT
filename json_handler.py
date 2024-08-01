import hashlib
import os
import pickle

import cv2
import numpy as np
import torch
import torchvision
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from torch.utils.data.dataset import Dataset


class JsonHandler(Dataset):
    def __init__(self, params_dict: dict, split_category: str):
        self.json_file_path = params_dict.get("json_file_path", "")
        self.delete_list = params_dict.get("delete_list", [])
        self.base_classes = params_dict.get("base_classes", [])
        self.out_classes = params_dict.get("out_classes", [])
        self.dataloader = params_dict.get("dataloader", False)
        self.resize = params_dict.get("resize", None)
        self.recalculate = params_dict.get("recalculate", False)
        self.delete_null = params_dict.get("delete_null", False)

        self.split_category = split_category

        if self.json_file_path and self.json_file_path[-1] != "/":
            self.json_file_path += "/"

        self.out_cl_hash = self.generate_class_id(self.out_classes)

        self.full_json_file_path = os.path.join(
            self.json_file_path, "annotations/instances_default.json"
        )
        self.coco = COCO(self.full_json_file_path)

        self.catIDs = self.coco.getCatIds()
        self.cats_to_classes = self.map_cats_to_classes()

        self.list_out_classes = ["фон"] + [cl["name"] for cl in self.out_classes]
        self.list_of_name_base_classes = ["фон"] + [
            cl["name"] for cl in self.base_classes
        ]

        self._train_list = None
        self._val_list = None
        self._all_img_list = None
        self._total_train = None
        self._total_val = None
        self._pixel_total_train = None
        self._pixel_total_val = None

    @property
    def train_list(self):
        if self._train_list is None:
            self._check_create_train_val_list()
        return self._train_list

    @property
    def val_list(self):
        if self._val_list is None:
            self._check_create_train_val_list()
        return self._val_list

    @property
    def all_img_list(self):
        if self._all_img_list is None:
            self._check_create_train_val_list()
        return self._all_img_list

    @property
    def total_train(self):
        if self._total_train is None:
            self._check_create_weight()
        return self._total_train

    @property
    def total_val(self):
        if self._total_val is None:
            self._check_create_weight()
        return self._total_val

    @property
    def pixel_total_train(self):
        if self._pixel_total_train is None:
            self._check_create_weight()
        return self._pixel_total_train

    @property
    def pixel_total_val(self):
        if self._pixel_total_val is None:
            self._check_create_weight()
        return self._pixel_total_val

    def map_cats_to_classes(self):
        cats_to_classes = {}
        cats = self.coco.loadCats(self.catIDs)
        for base_class in self.base_classes:
            for cat in cats:
                if base_class["name"] == cat["name"]:
                    cats_to_classes[cat["id"]] = base_class["id"]
        return cats_to_classes

    def getImgIds_all_cats(self, imgIDs, catIDs):
        ids_list = [
            self.coco.getImgIds(imgIds=imgIDs, catIds=catID) for catID in catIDs
        ]
        return list(set(sum(ids_list, [])))

    def __len__(self):
        return len(self.train_list) + len(self.val_list)

    def generate_class_id(self, classes):
        my_hash = 0
        for cl in classes:
            my_str = (
                str(cl["id"])
                + cl["name"]
                + str(cl["summable_masks"])
                + str(cl["subtractive_masks"])
            ).encode("utf-8")
            my_hash += int.from_bytes(hashlib.sha256(my_str).digest(), "little")
        return str(my_hash)

    def _check_create_train_val_list(self):
        train_list_path = os.path.join(
            self.json_file_path, f"train_list_{self.out_cl_hash}.pickle"
        )
        val_list_path = os.path.join(
            self.json_file_path, f"val_list_{self.out_cl_hash}.pickle"
        )
        all_img_list_path = os.path.join(
            self.json_file_path, f"all_img_list_{self.out_cl_hash}.pickle"
        )

        if (
            not self.recalculate
            and os.path.exists(train_list_path)
            and os.path.exists(val_list_path)
            and os.path.exists(all_img_list_path)
        ):
            with open(train_list_path, "rb") as f:
                self._train_list = pickle.load(f)
            with open(val_list_path, "rb") as f:
                self._val_list = pickle.load(f)
            with open(all_img_list_path, "rb") as f:
                self._all_img_list = pickle.load(f)
        else:
            self._generate_train_val_lists()

    def _generate_train_val_lists(self):
        imgIds = self.coco.getImgIds()

        if self._train_list is None:
            self._train_list = []
        if self._val_list is None:
            self._val_list = []
        if self._all_img_list is None:
            self._all_img_list = []

        for img_id in imgIds:
            anns_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.catIDs)
            anns = self.coco.loadAnns(anns_ids)

            if self.delete_null and len(anns) == 0:
                continue

            save = True
            for ann in anns:
                if (
                    self.coco.loadCats(ann["category_id"])[0]["name"]
                    in self.delete_list
                ):
                    save = False
                    break

            if save:
                if self.split_category == "val":
                    self._val_list.append(img_id)
                else:
                    self._train_list.append(img_id)
                self._all_img_list.append(img_id)

        self._save_lists()

    def _save_lists(self):
        train_list_path = os.path.join(
            self.json_file_path, f"train_list_{self.out_cl_hash}.pickle"
        )
        val_list_path = os.path.join(
            self.json_file_path, f"val_list_{self.out_cl_hash}.pickle"
        )
        all_img_list_path = os.path.join(
            self.json_file_path, f"all_img_list_{self.out_cl_hash}.pickle"
        )

        with open(train_list_path, "wb") as f:
            pickle.dump(self._train_list, f)
        with open(val_list_path, "wb") as f:
            pickle.dump(self._val_list, f)
        with open(all_img_list_path, "wb") as f:
            pickle.dump(self._all_img_list, f)

    def _check_create_weight(self):
        total_train_path = os.path.join(
            self.json_file_path, f"TotalTrain_{self.out_cl_hash}.pickle"
        )
        total_val_path = os.path.join(
            self.json_file_path, f"TotalVal_{self.out_cl_hash}.pickle"
        )
        pixel_total_train_path = os.path.join(
            self.json_file_path, f"pixel_TotalTrain_{self.out_cl_hash}.pickle"
        )
        pixel_total_val_path = os.path.join(
            self.json_file_path, f"pixel_TotalVal_{self.out_cl_hash}.pickle"
        )

        if (
            not self.recalculate
            and os.path.exists(total_train_path)
            and os.path.exists(total_val_path)
            and os.path.exists(pixel_total_train_path)
            and os.path.exists(pixel_total_val_path)
        ):
            with open(total_train_path, "rb") as f:
                self._total_train = pickle.load(f)
            with open(total_val_path, "rb") as f:
                self._total_val = pickle.load(f)
            with open(pixel_total_train_path, "rb") as f:
                self._pixel_total_train = pickle.load(f)
            with open(pixel_total_val_path, "rb") as f:
                self._pixel_total_val = pickle.load(f)

        else:
            self._calculate_weights()

    def _calculate_weights(self):
        len_out_cls = len(self.list_out_classes)
        self._total_train, self._total_val = (
            np.zeros(len_out_cls),
            np.zeros(len_out_cls),
        )
        self._pixel_total_train, self._pixel_total_val = (
            np.zeros(len_out_cls),
            np.zeros(len_out_cls),
        )

        for img_id in self.train_list:
            result = self.__getitem__(img_id)
            mask = result["masks"]

            for i in range(len_out_cls):
                self._total_train[i] += mask[i].max().item()
                self._pixel_total_train[i] += mask[i].sum().item()

        for img_id in self.val_list:
            result = self.__getitem__(img_id)
            mask = result["masks"]

            for i in range(len_out_cls):
                self._total_val[i] += mask[i].max().item()
                self._pixel_total_val[i] += mask[i].sum().item()

        self._save_weights()

    def _save_weights(self):
        total_train_path = os.path.join(
            self.json_file_path, f"TotalTrain_{self.out_cl_hash}.pickle"
        )
        total_val_path = os.path.join(
            self.json_file_path, f"TotalVal_{self.out_cl_hash}.pickle"
        )
        pixel_total_train_path = os.path.join(
            self.json_file_path, f"pixel_TotalTrain_{self.out_cl_hash}.pickle"
        )
        pixel_total_val_path = os.path.join(
            self.json_file_path, f"pixel_TotalVal_{self.out_cl_hash}.pickle"
        )

        with open(total_train_path, "wb") as f:
            pickle.dump(self._total_train, f)
        with open(total_val_path, "wb") as f:
            pickle.dump(self._total_val, f)
        with open(pixel_total_train_path, "wb") as f:
            pickle.dump(self._pixel_total_train, f)
        with open(pixel_total_val_path, "wb") as f:
            pickle.dump(self._pixel_total_val, f)

    def to_out_classes(self, mask):
        size = np.shape(mask)
        new_mask = np.zeros((len(self.out_classes) + 1, size[1], size[2]))
        new_mask[0] = 1

        for out_class in self.out_classes:
            for i in out_class["summable_masks"]:
                new_mask[out_class["id"], :, :][mask[i, :, :] == 1] = 1
            for i in out_class["subtractive_masks"]:
                new_mask[out_class["id"], :, :][mask[i, :, :] == 1] = 0
            new_mask[0][new_mask[out_class["id"], :, :] == 1] = 0

        return new_mask

    def load_image(self, img_id):
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.json_file_path, "images", img_info["file_name"])
        return cv2.imread(img_path)

    def load_annotations(self, img_id):
        anns_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.catIDs)
        anns = self.coco.loadAnns(anns_ids)
        return anns

    # было так но по ходу это не совсем верно, потому что если у меня segmentation имеет два
    # подсписка [[], []] то есть есть две области одного класса на фото, то эта функция создаст
    # картинку не (512, 512) а (512, 512, 2), а это ошибка
    # def process_mask(self, ann, image_height, image_width):
    #     class_idx = self.cats_to_classes[ann["category_id"]]
    #     rle = maskUtils.frPyObjects(ann["segmentation"], image_height, image_width)
    #     mask = maskUtils.decode(rle)
    #     return class_idx, mask

    def process_mask(self, ann, image_height, image_width):
        class_idx = self.cats_to_classes[ann["category_id"]]
        rles = maskUtils.frPyObjects(ann["segmentation"], image_height, image_width)

        # Создаем пустую маску для текущей аннотации
        combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        
        # Если rles это не список, делаем его списком
        if not isinstance(rles, list):
            rles = [rles]
        
        for rle in rles:
            mask = maskUtils.decode(rle)
            
            if len(mask.shape) == 3:
                mask = np.max(mask, axis=2)
            
            # Добавляем текущую маску к общей маске
            combined_mask = np.maximum(combined_mask, mask)

        return class_idx, combined_mask

    # так было ниже сделаю для флоренции
    # def __getitem__(self, idx, contours=False):
    #     img_info = self.coco.loadImgs(idx)[0]
    #     image = self.load_image(idx)
    #     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #     anns = self.load_annotations(idx)
    #     image_height = img_info["height"]
    #     image_width = img_info["width"]
    
    #     mask = np.zeros(
    #         (len(self.catIDs) - len(self.delete_list) + 1, image_height, image_width)
    #     )
    
    #     for ann in anns:
    #         if ann["category_id"] not in self.delete_list:
    #             class_idx, mask_instance = self.process_mask(
    #                 ann, image_height, image_width
    #             )
    #             mask_instance = np.squeeze(mask_instance)
    #             mask[class_idx] = np.maximum(mask[class_idx], mask_instance)
    
    #     mask = self.to_out_classes(mask)
    
    #     if self.resize and not contours:
    #         image = torch.unsqueeze(torch.tensor(gray_image), 0)
    #         image = torchvision.transforms.functional.resize(
    #             image, self.resize, antialias=True
    #         )
    #         mask = torchvision.transforms.functional.resize(
    #             torch.tensor(mask), self.resize, antialias=True
    #         )
    
    #         if not self.dataloader:
    #             image = torch.unsqueeze(image, 0)
    #             image = (image - image.min()) / (image.max() - image.min() + 1e-7)
    #             mask = torch.unsqueeze(mask, 0)
    #             rgb_image = cv2.resize(image.numpy().squeeze(), self.resize)
    #             return image.float(), mask.long(), rgb_image
    
    #         image = (image - image.min()) / (image.max() - image.min() + 1e-7)
    
    #         task_prompt = "<REFERRING_EXPRESSION_SEGMENTATION>"
    
    #         result = {
    #             "images": image.float(),
    #             "masks": mask.long(),
    #             "labels": torch.amax(mask, dim=(-1, -2)),
    #             "values": torch.sum(mask, (-1, -2)),
    #             "task_prompt": task_prompt,
    #         }
    
    #         # было так, но для флоренции меняю
    #         return result
    
    #         # task_prompt = "<REFERRING_EXPRESSION_SEGMENTATION>"
    #         # return task_prompt, image.float(), mask.long()
    
    #         # image, annotation = self.dataset[idx]
    #         # bbox = annotation["bbox"]
    #         # category_id = annotation["category_id"]
    #         # segmentation = annotation["segmentation"]
    #         # return image, bbox, category_id, segmentation
    
    #     else:
    #         return gray_image, mask
    

    # florence
    def __getitem__(self, idx, contours=False):
        img_info = self.coco.loadImgs(idx)[0]
        image = self.load_image(idx)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        anns = self.load_annotations(idx)
        image_height = img_info["height"]
        image_width = img_info["width"]

        mask = np.zeros(
            (len(self.catIDs) - len(self.delete_list) + 1, image_height, image_width)
        )

        bboxes = []
        category_ids = []
        segmentations = []

        for ann in anns:
            if ann["category_id"] not in self.delete_list:
                class_idx, mask_instance = self.process_mask(
                    ann, image_height, image_width
                )
                mask_instance = np.squeeze(mask_instance)
                mask[class_idx] = np.maximum(mask[class_idx], mask_instance)

                # Find bounding box for mask
                contours, _ = cv2.findContours(
                    mask_instance.astype(np.uint8),
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    x2 = x + w
                    y2 = y + h
                    # Normalize and scale bounding box
                    box = [
                        x / image_width,
                        y / image_height,
                        x2 / image_width,
                        y2 / image_height,
                    ]
                    box = [coord * 1000 for coord in box]
                    bboxes.append(box)

                category_ids.append(ann["category_id"])
                segmentations.append(ann["segmentation"])

        mask = self.to_out_classes(mask)

        if self.resize:# and not contours:
            image = torch.unsqueeze(torch.tensor(gray_image), 0)
            image = torchvision.transforms.functional.resize(
                image, self.resize, antialias=True
            )
            mask = torchvision.transforms.functional.resize(
                torch.tensor(mask), self.resize, antialias=True
            )

            if not self.dataloader:
                image = torch.unsqueeze(image, 0)
                image = (image - image.min()) / (image.max() - image.min() + 1e-7)
                mask = torch.unsqueeze(mask, 0)
                rgb_image = cv2.resize(image.numpy().squeeze(), self.resize)
                return image.float(), mask.long(), rgb_image

            image = (image - image.min()) / (image.max() - image.min() + 1e-7)

            task_prompt = "<OD>"

            result = {
                "image_id": img_info["id"],
                "image_file": img_info["file_name"],
                "images": image.float(),
                "masks": mask.long(),
                "labels": torch.amax(mask, dim=(-1, -2)),
                "values": torch.sum(mask, (-1, -2)),
                "task_prompt": task_prompt,
                "bboxes": bboxes,
                "category_ids": category_ids,
                "segmentations": segmentations,
            }

            # было так, но для флоренции меняю
            return result

            # task_prompt = "<REFERRING_EXPRESSION_SEGMENTATION>"
            # return task_prompt, image.float(), mask.long()

            # image, annotation = self.dataset[idx]
            # bbox = annotation["bbox"]
            # category_id = annotation["category_id"]
            # segmentation = annotation["segmentation"]
            # return image, bbox, category_id, segmentation

        else:
            return gray_image, mask
