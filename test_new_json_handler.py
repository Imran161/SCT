import cv2
import os
import numpy as np
import torch
import pickle
import torchvision
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


class JsonHandler:
    def __init__(self, params_dict):
        self.json_file_path = params_dict.get("json_file_path", "")
        self.delete_list = params_dict.get("delete_list", [])
        self.base_classes = params_dict.get("base_classes", [])
        self.out_classes = params_dict.get("out_classes", [])
        self.dataloader = params_dict.get("dataloader", False)
        self.resize = params_dict.get("resize")
        self.recalculate = params_dict.get("recalculate", False)
        self.delete_null = params_dict.get("delete_null", False)
        self.train_val_probs = params_dict.get("train_val_probs")

        if self.json_file_path and self.json_file_path[-1] != "/":
            self.json_file_path += "/"

        self.out_cl_hash = self.generate_class_id(self.out_classes)
        full_json_file_path = os.path.join(
            self.json_file_path, "annotations/instances_default.json"
        )
        self.coco = COCO(full_json_file_path)
        self.catIDs = self.coco.getCatIds()
        self.ids_idx_dict = {}
        self.list_of_name_out_classes = ["фон"]
        self.list_of_name_base_classes = ["фон"]

        for segmentation_class in self.base_classes:
            self.list_of_name_base_classes.append(segmentation_class["name"])
        for segmentation_class in self.out_classes:
            self.list_of_name_out_classes.append(segmentation_class["name"])

        self.cats_to_classes = {}
        cats = self.coco.loadCats(self.catIDs)

        for cl in self.base_classes:
            for cat in cats:
                if cl["name"] == cat["name"]:
                    self.cats_to_classes[cat["id"]] = cl["id"]

        self.check_create_train_val_list(self.recalculate)
        self.check_create_weight(self.recalculate)

        self.colors = [
            ((251, 206, 177), "Абрикосовым"),
            ((127, 255, 212), "Аквамариновым"),
            ((255, 36, 0), "Алым"),
            ((153, 102, 204), "Аметистовым"),
            ((153, 0, 102), "Баклажановым"),
            ((48, 213, 200), "Бирюзовым"),
            ((152, 251, 152), "Бледно зеленым"),
            ((213, 113, 63), "Ванильным"),
            ((100, 149, 237), "Васильковым"),
            ((34, 139, 34), "Зелёный лесной"),
            ((0, 0, 255), "Синий"),
            ((75, 0, 130), "Индиго"),
            ((255, 0, 255), "Чёрный"),
            ((0, 51, 153), "Маджента"),
            ((65, 105, 225), "Королевский синий"),
            ((255, 255, 0), "Жёлтый"),
            ((255, 69, 0), "Оранжево-красный"),
            ((255, 0, 0), "Темно синим"),
            ((0, 51, 153), "Красный"),
            ((255, 215, 0), "Золотой"),
            ((250, 128, 114), "Лососевый"),
            ((255, 99, 71), "Томатный"),
            ((255, 215, 0), "Золотой"),
            ((0, 139, 139), "Тёмный циан"),
            ((0, 255, 255), "Морская волна"),
        ]

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

    def check_create_train_val_list(self, recalculate):
        try:
            train_list = open(
                os.path.join(
                    self.json_file_path, f"train_list_{self.out_cl_hash}.pickle"
                ),
                "rb",
            )
            val_list = open(
                os.path.join(
                    self.json_file_path, f"val_list_{self.out_cl_hash}.pickle"
                ),
                "rb",
            )
            all_img_list = open(
                os.path.join(
                    self.json_file_path, f"all_img_list_{self.out_cl_hash}.pickle"
                ),
                "rb",
            )

            self.train_list = pickle.load(train_list)
            self.val_list = pickle.load(val_list)
            self.all_img_list = pickle.load(all_img_list)

            if recalculate:  # != False было
                raise ValueError("stuff is not in content")

        except FileNotFoundError:
            imgIds = self.coco.getImgIds()
            print("готовые данные не обраружены или включена рекалькуляция")
            train_list = []
            val_list = []
            all_img_list = []

            for img_index in imgIds:
                anns_ids = self.coco.getAnnIds(imgIds=img_index, catIds=self.catIDs)
                anns = self.coco.loadAnns(anns_ids)
                # print("anns", anns)
                save = True
                if len(anns) == 0 and self.delete_null:  # == True:
                    save = False
                    print("отсутствуют какие либо метки:", img_index)
                for ann in anns:
                    cat = self.coco.loadCats(ann["category_id"])[0]
                    if cat["name"] in self.delete_list:
                        save = False
                        non_support_class = cat["name"]
                        print(
                            f"недопустимое значение метки:{non_support_class}",
                            img_index,
                        )
                # print("save", save)
                if save:
                    # try:
                    x = random.randint(1, 100)
                    if x >= self.train_val_probs:  # 80
                        val_list.append(img_index)
                    else:
                        train_list.append(img_index)
                        self.train_list = train_list
                    all_img_list.append(img_index)
                    # except:
                    #     pass
                    #     # print("Тут файл без размет

    def check_create_weight(self, recalculate):
        try:
            TotalTrain = open(
                os.path.join(
                    self.json_file_path, f"TotalTrain_{self.out_cl_hash}.pickle"
                ),
                "rb",
            )
            TotalVal = open(
                os.path.join(
                    self.json_file_path, f"TotalVal_{self.out_cl_hash}.pickle"
                ),
                "rb",
            )
            pixel_TotalTrain = open(
                os.path.join(
                    self.json_file_path, f"pixel_TotalTrain_{self.out_cl_hash}.pickle"
                ),
                "rb",
            )
            pixel_TotalVal = open(
                os.path.join(
                    self.json_file_path, f"pixel_TotalVal_{self.out_cl_hash}.pickle"
                ),
                "rb",
            )

            self.TotalTrain = pickle.load(TotalTrain)
            self.TotalVal = pickle.load(TotalVal)
            self.pixel_TotalTrain = pickle.load(pixel_TotalTrain)
            self.pixel_TotalVal = pickle.load(pixel_TotalVal)

            if recalculate:  # != False:
                raise ValueError("stuff is not in content")

        except FileNotFoundError:
            print("готовые веса не обраружены или включена рекалькуляция")

            noc = len(self.list_of_name_out_classes)
            TotalTrain = np.zeros(noc)
            TotalVal = np.zeros(noc)

            pixel_TotalTrain = np.zeros(noc)
            pixel_TotalVal = np.zeros(noc)

            for i in self.train_list:
                result = self.__getitem__(i)
                # print(type(result))
                # print("result[images]", result["images"])
                image, mask = result["images"], result["masks"]
                for j in range(noc):
                    TotalTrain[j] += mask[j].max().item()
                    pixel_TotalTrain[j] += mask[j].sum().item()

            for i in self.val_list:
                result = self.__getitem__(i)
                image, mask = result["images"], result["masks"]
                for j in range(noc):
                    TotalVal[j] += mask[j].max().item()
                    pixel_TotalVal += mask[j].sum().item()

            self.TotalTrain = TotalTrain
            self.TotalVal = TotalVal
            self.pixel_TotalTrain = pixel_TotalTrain
            self.pixel_TotalVal = pixel_TotalVal

            with open(
                os.path.join(
                    self.json_file_path, f"TotalTrain_{self.out_cl_hash}.pickle"
                ),
                "wb",
            ) as Total:
                pickle.dump(self.TotalTrain, Total)
            with open(
                os.path.join(
                    self.json_file_path, f"TotalVal_{self.out_cl_hash}.pickle"
                ),
                "wb",
            ) as Total:
                pickle.dump(self.TotalVal, Total)
            with open(
                os.path.join(
                    self.json_file_path, f"pixel_TotalTrain_{self.out_cl_hash}.pickle"
                ),
                "wb",
            ) as pix_Total:
                pickle.dump(self.pixel_TotalTrain, pix_Total)
            with open(
                os.path.join(
                    self.json_file_path, f"pixel_TotalVal_{self.out_cl_hash}.pickle"
                ),
                "wb",
            ) as pix_Total:
                pickle.dump(self.pixel_TotalVal, pix_Total)

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

    def __getitem__(self, idx, contures=False):
        images_description = self.coco.loadImgs(idx)[0]
        image_path = os.path.join(
            self.json_file_path, "images", images_description["file_name"]
        )
        rgb_image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        anns_ids = self.coco.getAnnIds(imgIds=idx, catIds=self.catIDs, iscrowd=None)
        anns = self.coco.loadAnns(anns_ids)
        mask = np.zeros(
            (
                len(self.catIDs) - len(self.delete_list) + 1,
                int(images_description["height"]),
                int(images_description["width"]),
            )
        )

        for ann in anns:
            cat = self.coco.loadCats(ann["category_id"])[0]
            if cat["name"] not in self.delete_list:
                class_mask = self.coco.annToMask(ann)
                class_idx = self.cats_to_classes[ann["category_id"]]
                mask[class_idx][class_mask == 1] = 1

        mask = self.to_out_classes(mask)

        if self.resize is not None and not contures:  # без not и == False было
            image = torch.unsqueeze(torch.tensor(gray_image), 0)
            image = torchvision.transforms.functional.resize(image, (self.resize))
            mask = torchvision.transforms.functional.resize(
                torch.tensor(mask), (self.resize)
            )
            if not self.dataloader:
                image = torch.unsqueeze(image, 0)
                image = (image - image.min()) / (image.max() - image.min() + 0.00000001)
                mask = torch.unsqueeze(mask, 0)
                rgb_image = cv2.resize(rgb_image, (self.resize))
                image = image.float()
                mask = mask.long()
                return image, mask, rgb_image

            image = (image - image.min()) / (image.max() - image.min() + 0.00000001)
            image = image.float()
            mask = mask.long()
            result = {}
            result["images"] = image
            result["masks"] = mask
            result["labels"] = torch.amax(mask, dim=(-1, -2))
            result["values"] = torch.sum(mask, (-1, -2))
            result["rgb_image"] = rgb_image

            return result
        else:
            # этот иф вызывается для индекса картики который хочу нарисовать, для остальных верхний вызывается
            # ну это потому что в шоу контрс getitem вызывается от contures=True так что все норм
            # print("мы внизу")
            return (gray_image, mask, rgb_image)
