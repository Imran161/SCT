# а в этом файле у меня будут создаваться файлы jsonl в которых каждая категория записана несколько раз синонимами

# этот код еще не проверял

import os
import json
import cv2
import numpy as np
import urllib.parse
from tqdm import tqdm
from pycocotools import mask as maskUtils
import zipfile 

class DataProcessor:
    def __init__(self, src_folder, dst_folder, task_type, synonyms_file, augment=False):
        self.src_folder = src_folder
        self.dst_folder = dst_folder
        self.task_type = task_type
        self.augment = augment
        self.synonyms = self.load_synonyms(synonyms_file)
        os.makedirs(dst_folder, exist_ok=True)
        
        self.processed_dirs_file = "/home/imran-nasyrov/SCT/florence2/processed_dirs.txt"
        # Загружаем список уже обработанных папок
        if os.path.exists(self.processed_dirs_file):
            with open(self.processed_dirs_file, "r") as f:
                self.processed_dirs = set(line.strip() for line in f)
        else:
            self.processed_dirs = set()

    def load_synonyms(self, filepath):
        # Чтение файла с синонимами
        with open(filepath, "r", encoding="utf-8") as f:
            synonym_dict = {}
            for line in f:
                key, values = line.strip().split(":")
                key = key.strip()
                values = [val.strip().strip('["]') for val in values.split(",")]
                synonym_dict[key] = values
        return synonym_dict

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
            resized_boxes.append([x1, y1, x2, y2])

        return resized_image, resized_boxes

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
                # if not os.path.isdir(task_path):
                #     continue
                
                # Если это ZIP-файл, распаковываем его
                if zipfile.is_zipfile(task_path):
                    print(f"Распаковываем {task_folder}")
                    with zipfile.ZipFile(task_path, 'r') as zip_ref:
                        extract_path = os.path.join(self.src_folder, task_folder.replace(".zip", ""))
                        zip_ref.extractall(extract_path)
                        task_path = extract_path  # Обновляем путь для дальнейшей обработки
                
                # Теперь проверяем, является ли это папкой
                if not os.path.isdir(task_path):
                    print(f"{task_folder} не является папкой, пропускаем.")
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
                        
                        # Проверяем, нужно ли исключить этот класс
                        if category_name in self.synonyms and "удалить" in self.synonyms[category_name]:
                            continue

                        # Получаем маску и контуры объекта
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

                    # Аугментация данных (если включена)
                    if self.augment:
                        transform = self.get_augmentation_pipeline()
                        augmented = transform(image=resized_image, bboxes=resized_boxes)
                        resized_image = augmented['image']
                        resized_boxes = augmented['bboxes']

                    dst_image_path = os.path.join(
                        dst_task_path, "images", image_filename
                    )
                    os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
                    cv2.imwrite(dst_image_path, resized_image)

                    # Обрабатываем каждый объект несколько раз с использованием синонимов
                    if self.task_type == "CAPTION_TO_PHRASE_GROUNDING":
                        class_to_boxes = {}

                        for i, box in enumerate(resized_boxes):
                            if i >= len(all_categories):
                                break
                            category_name = all_categories[i]

                            if category_name not in class_to_boxes:
                                class_to_boxes[category_name] = []

                            class_to_boxes[category_name].append(f"<loc_{box[0]}><loc_{box[1]}><loc_{box[2]}><loc_{box[3]}>")

                        for category_name, box_locations in class_to_boxes.items():
                            if category_name in self.synonyms:
                                for synonym in self.synonyms[category_name]:
                                    suffix = synonym + "".join(box_locations)
                                    jsonl_data.append(
                                        self.create_jsonl_entry(
                                            image_filename,
                                            self.task_type + " " + synonym,
                                            suffix,
                                        )
                                    )

                # Запись в jsonl файл
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
    src_folder = "/home/imran-nasyrov/cvat/"  # Путь к исходной папке с архивами
    dst_folder = "/home/imran-nasyrov/cvat_sinonyms/"  # Путь к папке для jsonl файлов
    task_type = "CAPTION_TO_PHRASE_GROUNDING"  # Тип задачи
    synonyms_file = "/home/imran-nasyrov/SCT/florence2/synonyms.txt"  # Файл с синонимами

    augment = False  # Аугментация отключена

    processor = DataProcessor(src_folder, dst_folder, task_type, synonyms_file, augment=augment)
    processor.process_directory()
