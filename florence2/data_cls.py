import json
import os

import cv2
import numpy as np
import torch
from pycocotools import mask as maskUtils
from tqdm import tqdm
import os
import zipfile
import json
import urllib.parse
from googletrans import Translator
from tqdm import tqdm

# from ..transforms import SegTransform
import albumentations as A


# было
class old_DataProcessor:
    def __init__(self, src_folder, dst_folder, task_type, augment=False):
        self.src_folder = src_folder
        self.dst_folder = dst_folder
        self.task_type = task_type
        self.augment = augment
        os.makedirs(dst_folder, exist_ok=True)
        
        self.seg_transform = SegTransform() if augment else None

    def process_mask(self, ann, image_height, image_width):
        rles = maskUtils.frPyObjects(ann["segmentation"], image_height, image_width)
        combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)

        if not isinstance(rles, list):
            rles = [rles]

        for rle in rles:
            mask = maskUtils.decode(rle)
            if len(mask.shape) == 3:
                mask = np.max(mask, axis=2)
            combined_mask = np.maximum(combined_mask, mask)
        return combined_mask

    def resize_image_and_boxes(self, image, boxes, target_size):
        h, w = image.shape[:2]
        resized_image = cv2.resize(image, (target_size, target_size))
        scale_x = target_size / w
        scale_y = target_size / h
        resized_boxes = []

        for box in boxes:
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)

            scaled_x1 = int(x1 * 1000 / target_size)
            scaled_y1 = int(y1 * 1000 / target_size)
            scaled_x2 = int(x2 * 1000 / target_size)
            scaled_y2 = int(y2 * 1000 / target_size)

            resized_boxes.append([scaled_x1, scaled_y1, scaled_x2, scaled_y2])

        return resized_image, resized_boxes

    def resize_segmentation(self, segmentation, scale_x, scale_y):
        resized_segmentation = []
        for polygon in segmentation:
            resized_polygon = [
                [int(point[0] * scale_x * 1000), int(point[1] * scale_y * 1000)]
                for point in zip(polygon[0::2], polygon[1::2])
            ]
            resized_segmentation.append(resized_polygon)
        return resized_segmentation

    def create_jsonl_entry(self, image_filename, prefix, suffix):
        return {
            "image": image_filename,
            "prefix": prefix,
            "suffix": suffix,
        }

    def process_directory(self):
        with tqdm(
            total=len(os.listdir(self.src_folder)), desc="Processing directories"
        ) as pbar_dirs:
            for task_folder in os.listdir(self.src_folder):
                task_path = os.path.join(self.src_folder, task_folder)
                if not os.path.isdir(task_path):
                    continue

                dst_task_path = os.path.join(self.dst_folder, task_folder)
                os.makedirs(dst_task_path, exist_ok=True)

                annotations_path = os.path.join(
                    task_path, "annotations", "instances_default.json"
                )
                with open(annotations_path, "r") as f:
                    coco_data = json.load(f)

                annotations = coco_data["annotations"]
                images = coco_data["images"]
                categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
                jsonl_data = []

                for image_info in images:
                    image_id = image_info["id"]
                    image_filename = image_info["file_name"]
                    width = image_info["width"]
                    height = image_info["height"]

                    image_annotations = [
                        ann for ann in annotations if ann["image_id"] == image_id
                    ]

                    all_boxes = []
                    all_categories = []
                    all_segmentations = []

                    for ann in image_annotations:
                        if ann["category_id"] not in categories:
                            continue

                        category_name = categories[ann["category_id"]]
                        combined_mask = self.process_mask(ann, height, width)
                        contours, _ = cv2.findContours(
                            combined_mask.astype(np.uint8),
                            cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        if not contours:
                            continue

                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            x2 = x + w
                            y2 = y + h
                            all_boxes.append([x, y, x2, y2])
                            all_categories.append(category_name)
                            all_segmentations.append(ann["segmentation"])

                    image_path = os.path.join(task_path, "images", image_filename)
                    image = cv2.imread(image_path)
                    if image is None:
                        continue

                    resized_image, resized_boxes = self.resize_image_and_boxes(
                        image, all_boxes, 1024
                    )
                    scale_x = 1024 / width
                    scale_y = 1024 / height
                    resized_segmentations = [
                        self.resize_segmentation(seg, scale_x, scale_y)
                        for seg in all_segmentations
                    ]

                    # Аугментация данных
                    if self.augment:
                        image_tensor = (
                            torch.from_numpy(resized_image).permute(2, 0, 1).float()
                            / 255.0
                        )
                        mask_tensors = torch.zeros(
                            (len(resized_segmentations), 1024, 1024)
                        )
                        for i, seg in enumerate(resized_segmentations):
                            for polygon in seg:
                                cv2.fillPoly(
                                    mask_tensors[i].numpy(), [np.array(polygon)], 1
                                )
                                
                        aug_image, aug_masks = self.seg_transform.apply_transform(
                            image_tensor, mask_tensors
                        )
                        
                        resized_image = (
                            aug_image.permute(1, 2, 0).numpy() * 255
                        ).astype(np.uint8)
                        resized_segmentations = [
                            cv2.findContours(
                                mask.numpy().astype(np.uint8),
                                cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE,
                            )[0]
                            for mask in aug_masks
                        ]

                    dst_image_path = os.path.join(
                        dst_task_path, "images", image_filename
                    )
                    os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
                    cv2.imwrite(dst_image_path, resized_image)

                    if self.task_type == "OD":
                        suffix = ""
                        for i, box in enumerate(resized_boxes):
                            if i >= len(all_categories):
                                break
                            category_name = all_categories[i]
                            suffix += f"{category_name}<loc_{box[0]}><loc_{box[1]}><loc_{box[2]}><loc_{box[3]}>"
                        jsonl_data.append(
                            self.create_jsonl_entry(image_filename, "<OD>", suffix)
                        )

                    elif self.task_type == "CAPTION_TO_PHRASE_GROUNDING":
                        for i, box in enumerate(resized_boxes):
                            if i >= len(all_categories):
                                break
                            category_name = all_categories[i]
                            suffix = f"{category_name}<loc_{box[0]}><loc_{box[1]}><loc_{box[2]}><loc_{box[3]}>"
                            jsonl_data.append(
                                self.create_jsonl_entry(
                                    image_filename,
                                    "<CAPTION_TO_PHRASE_GROUNDING>",
                                    suffix,
                                )
                            )

                    elif self.task_type == "REFERRING_EXPRESSION_SEGMENTATION":
                        for i, segmentation in enumerate(resized_segmentations):
                            if i >= len(all_categories):
                                break
                            category_name = all_categories[i]
                            suffix = f"{category_name}<loc_{segmentation}>"
                            jsonl_data.append(
                                self.create_jsonl_entry(
                                    image_filename,
                                    "<REFERRING_EXPRESSION_SEGMENTATION>",
                                    suffix,
                                )
                            )

                jsonl_file_path = os.path.join(dst_task_path, "annotations.jsonl")
                with open(jsonl_file_path, "w") as f:
                    for entry in jsonl_data:
                        json.dump(entry, f)
                        f.write("\n")

                pbar_dirs.update(1)





# new


import albumentations as A
import cv2
import json
import numpy as np
import os
import torch
from pycocotools import mask as maskUtils
from tqdm import tqdm

processed_dirs_file = "/home/imran-nasyrov/SCT/florence2/processed_dirs.txt"

# Загружаем список уже обработанных папок
if os.path.exists(processed_dirs_file):
    with open(processed_dirs_file, "r") as f:
        processed_dirs = set(line.strip() for line in f)
else:
    processed_dirs = set()


import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DataProcessor:
    def __init__(self, src_folder, dst_folder, task_type, augment=False):
        self.src_folder = src_folder
        self.dst_folder = dst_folder
        self.task_type = task_type
        self.augment = augment
        os.makedirs(dst_folder, exist_ok=True)
        
        self.translator = Translator()
        self.translator.raise_exception = True  # Исправляем с 'raise_Exception'

        # self.transform = self.get_augmentation_pipeline() if augment else None
        
        self.processed_dirs_file = "/home/imran-nasyrov/SCT/florence2/processed_dirs.txt"
        # Загружаем список уже обработанных папок
        if os.path.exists(self.processed_dirs_file):
            with open(self.processed_dirs_file, "r") as f:
                self.processed_dirs = set(line.strip() for line in f)
        else:
            self.processed_dirs = set()
            

    def get_augmentation_pipeline(self):
        return A.Compose([
            A.RandomRotate90(),
            A.Flip(),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(p=0.2),
            A.HueSaturationValue(p=0.3),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


    def process_mask(self, ann, image_height, image_width):
        rles = maskUtils.frPyObjects(ann["segmentation"], image_height, image_width)
        combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)

        if not isinstance(rles, list):
            rles = [rles]

        for rle in rles:
            mask = maskUtils.decode(rle)
            if len(mask.shape) == 3:
                mask = np.max(mask, axis=2)
            combined_mask = np.maximum(combined_mask, mask)
        return combined_mask

    def resize_image_and_boxes(self, image, boxes, target_size):
        h, w = image.shape[:2]
        resized_image = cv2.resize(image, (target_size, target_size))
        scale_x = target_size / w
        scale_y = target_size / h
        resized_boxes = []

        for box in boxes:
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)

            scaled_x1 = int(x1 * 1000 / target_size)
            scaled_y1 = int(y1 * 1000 / target_size)
            scaled_x2 = int(x2 * 1000 / target_size)
            scaled_y2 = int(y2 * 1000 / target_size)

            resized_boxes.append([scaled_x1, scaled_y1, scaled_x2, scaled_y2])

        return resized_image, resized_boxes

    def resize_segmentation(self, segmentation, scale_x, scale_y):
        resized_segmentation = []
        for polygon in segmentation:
            resized_polygon = [
                [int(point[0] * scale_x * 1000), int(point[1] * scale_y * 1000)]
                for point in zip(polygon[0::2], polygon[1::2])
            ]
            resized_segmentation.append(resized_polygon)
        return resized_segmentation

    def create_jsonl_entry(self, image_filename, prefix, suffix):
        return {
            "image": image_filename,
            "prefix": prefix,
            "suffix": suffix,
        }
        
        
    def decode_and_translate_categories(self, categories):
        translated_categories = []
        for category in categories:
            # Декодирование имени категории с UTF-8
            decoded_name = urllib.parse.unquote(category["name"])

            # Проверка, что категория уже на английском (ASCII)
            if all(ord(char) < 128 for char in decoded_name):
                translated_categories.append(category)
                continue

            # Перевод на английский язык
            translated_name = self.translator.translate(decoded_name, src="ru", dest="en").text

            # Обновление имени категории
            category["name"] = translated_name
            translated_categories.append(category)

        return translated_categories


    def process_directory(self):
        with tqdm(
            total=len(os.listdir(self.src_folder)), desc="Processing directories"
        ) as pbar_dirs:
            for task_folder in os.listdir(self.src_folder):
                task_path = os.path.join(self.src_folder, task_folder)
                if not os.path.isdir(task_path):
                    continue
                
                # Пропускаем папки, которые уже были обработаны
                if task_folder in self.processed_dirs:
                    print(f"Папка {task_folder} уже обработана, пропускаем.")
                    pbar_dirs.update(1)
                    continue
                
                dst_task_path = os.path.join(self.dst_folder, task_folder)
                os.makedirs(dst_task_path, exist_ok=True)

                annotations_path = os.path.join(
                    task_path, "annotations", "instances_default.json"
                )
                with open(annotations_path, "r") as f:
                    coco_data = json.load(f)

                annotations = coco_data["annotations"]
                images = coco_data["images"]
                # было
                # categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
                
                # стало
                cat = coco_data["categories"]
                categories = self.decode_and_translate_categories(cat)
                categories = {cat["id"]: cat["name"] for cat in cat}
                # print("categories after", categories)
                
                # "categories": [{"id": 1, "name": "right_kidney_ID1", "supercategory": ""}, ... ]
                # break
                jsonl_data = []

                for image_info in images:
                    image_id = image_info["id"]
                    image_filename = image_info["file_name"]
                    width = image_info["width"]
                    height = image_info["height"]

                    image_annotations = [
                        ann for ann in annotations if ann["image_id"] == image_id
                    ]

                    all_boxes = []
                    all_categories = []
                    all_segmentations = []

                    for ann in image_annotations:
                        if ann["category_id"] not in categories:
                            continue

                        category_name = categories[ann["category_id"]]
                        combined_mask = self.process_mask(ann, height, width)
                        contours, _ = cv2.findContours(
                            combined_mask.astype(np.uint8),
                            cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        if not contours:
                            continue

                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            x2 = x + w
                            y2 = y + h
                            all_boxes.append([x, y, x2, y2])
                            all_categories.append(category_name)
                            all_segmentations.append(ann["segmentation"])

                    image_path = os.path.join(task_path, "images", image_filename)
                    image = cv2.imread(image_path)
                    if image is None:
                        continue

                    resized_image, resized_boxes = self.resize_image_and_boxes(
                        image, all_boxes, 1024
                    )
                    scale_x = 1024 / width
                    scale_y = 1024 / height
                    resized_segmentations = [
                        self.resize_segmentation(seg, scale_x, scale_y)
                        for seg in all_segmentations
                    ]

                    # print("resized_image", resized_image.shape)
                    # print("combined_mask", combined_mask.shape)
                    # Аугментация данных
                    
                   
                    if self.augment:
                        labels = all_categories
                        transform = self.get_augmentation_pipeline()
                        
                        augmented = transform(image=resized_image, bboxes=resized_boxes, labels=labels)
                        resized_image = augmented['image']
                        resized_boxes = augmented['bboxes']
                        
                        resized_boxes = [[int(coord) for coord in box] for box in resized_boxes]
                    
                        
                    dst_image_path = os.path.join(
                        dst_task_path, "images", image_filename
                    )
                    os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
                    cv2.imwrite(dst_image_path, resized_image)

                    if self.task_type == "OD":
                        suffix = ""
                        for i, box in enumerate(resized_boxes):
                            if i >= len(all_categories):
                                break
                            category_name = all_categories[i]
                            suffix += f"{category_name}<loc_{box[0]}><loc_{box[1]}><loc_{box[2]}><loc_{box[3]}>"
                        jsonl_data.append(
                            self.create_jsonl_entry(
                                image_filename, 
                                self.task_type, 
                                suffix)
                        )

                    # тут для каждого объекта своя строка, а надо для одного класса на картинке в одну строку писать 
                    # elif self.task_type == "CAPTION_TO_PHRASE_GROUNDING":
                    #     for i, box in enumerate(resized_boxes):
                    #         if i >= len(all_categories):
                    #             break
                    #         category_name = all_categories[i]
                    #         suffix = f"{category_name}<loc_{box[0]}><loc_{box[1]}><loc_{box[2]}><loc_{box[3]}>"
                    #         jsonl_data.append(
                    #             self.create_jsonl_entry(
                    #                 image_filename,
                    #                 self.task_type + " " + category_name,
                    #                 suffix,
                    #             )
                    #         )


                    elif self.task_type == "CAPTION_TO_PHRASE_GROUNDING":
                        class_to_boxes = {}

                        for i, box in enumerate(resized_boxes):
                            if i >= len(all_categories):
                                break
                            category_name = all_categories[i]

                            if category_name not in class_to_boxes:
                                class_to_boxes[category_name] = []

                            class_to_boxes[category_name].append(f"<loc_{box[0]}><loc_{box[1]}><loc_{box[2]}><loc_{box[3]}>")

                        for category_name, box_locations in class_to_boxes.items():
                            suffix = category_name + "".join(box_locations)
                            jsonl_data.append(
                                self.create_jsonl_entry(
                                    image_filename,
                                    self.task_type + " " + category_name,
                                    suffix,
                                )
                            )


                    # надо исправить
                    #####################################
                    elif self.task_type == "REFERRING_EXPRESSION_SEGMENTATION":
                        for i, segmentation in enumerate(resized_segmentations):
                            if i >= len(all_categories):
                                break
                            category_name = all_categories[i]
                            suffix = f"{category_name}<loc_{segmentation}>" # вот тут 
                            jsonl_data.append(
                                self.create_jsonl_entry(
                                    image_filename,
                                    self.task_type,
                                    suffix,
                                )
                            )

                jsonl_file_path = os.path.join(dst_task_path, "annotations.jsonl")
                with open(jsonl_file_path, "w") as f:
                    for entry in jsonl_data:
                        json.dump(entry, f)
                        f.write("\n")

                # Записываем папку в файл обработанных папок
                with open(self.processed_dirs_file, "a") as f:
                    f.write(task_folder + "\n")
                self.processed_dirs.add(task_folder)
                
                pbar_dirs.update(1)





    


if __name__ == "__main__":
    src_folder = "/home/imran-nasyrov/cvat_lung_unzip/" #"sinusite_json_data"
    dst_folder = "/home/imran-nasyrov/cvat_lung_phrase" # test_sinusite_jsonl
    task_type = "CAPTION_TO_PHRASE_GROUNDING"  # "OD" или "CAPTION_TO_PHRASE_GROUNDING" или "REFERRING_EXPRESSION_SEGMENTATION" 
    
    #######
    augment = False  # короче не очень трансформы работают что то
    ########
    

    processor = DataProcessor(src_folder, dst_folder, task_type, augment=augment)
    processor.process_directory()


# надо так
# questions: CAPTION_TO_PHRASE_GROUNDING Right frontal sinus (external contour)
# answer: Right frontal sinus (external contour)<loc_419><loc_332><loc_505><loc_419>

# и сделать одного класса вместе 

# выбирать задание на лету, возможно вместе
# обычный лосс и bce перезапустить