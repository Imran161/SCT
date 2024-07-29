import base64
import json
import os
import random
from typing import Any, Dict, List, Tuple
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from PIL import Image, ImageDraw, ImageFont
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForImageSegmentation,
    AutoProcessor,
    SegformerConfig,
    SegformerFeatureExtractor,
    SegformerForSemanticSegmentation,
    get_scheduler,
)

from coco_classes import sinusite_base_classes, sinusite_pat_classes_3
from coco_dataloaders import FLORENCE_COCODataLoader, SINUSITE_COCODataLoader
from metrics import DetectionMetrics
from sct_val import test_model
from transforms import SegTransform
from utils import ExperimentSetup, iou_metric, save_best_metrics_to_csv, set_seed

set_seed(64)


import os
import json
import shutil
from pathlib import Path
import numpy as np
import cv2
from pycocotools import mask as maskUtils


# тут боксы не все 

# # Указать пути к папкам
# src_folder = 'sinusite_json_data'
# dst_folder = 'sinusite_jsonl'

# # Создать целевую папку, если она не существует
# os.makedirs(dst_folder, exist_ok=True)

# # Словарь для перевода id в имя класса на английском языке
# class_id_to_name = {
#     1: "Right maxillary sinus (outer contour)",
#     2: "Left maxillary sinus (outer contour)",
#     3: "Left frontal sinus (outer contour)",
#     4: "Right frontal sinus (outer contour)",
#     5: "Right maxillary sinus (inner void boundary)",
#     6: "Left maxillary sinus (inner void boundary)",
#     7: "Left frontal sinus (inner void boundary)",
#     8: "Right frontal sinus (inner void boundary)",
#     9: "Reduction of pneumatization of paranasal sinuses",
#     10: "Horizontal fluid-air level",
#     11: "Absence of pneumatization of paranasal sinuses",
#     12: "Other pathology",
#     13: "Inscription"
# }

# # Функция для извлечения ограничивающих рамок из маски
# def process_mask(ann, image_height, image_width):
#     rles = maskUtils.frPyObjects(ann["segmentation"], image_height, image_width)

#     # Создаем пустую маску для текущей аннотации
#     combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)

#     # Если rles это не список, делаем его списком
#     if not isinstance(rles, list):
#         rles = [rles]

#     for rle in rles:
#         mask = maskUtils.decode(rle)

#         if len(mask.shape) == 3:
#             mask = np.max(mask, axis=2)

#         # Добавляем текущую маску к общей маске
#         combined_mask = np.maximum(combined_mask, mask)

#     return combined_mask

# # Функция для изменения размера изображения и корректировки bounding boxes
# def resize_image_and_boxes(image, boxes, target_size):
#     h, w = image.shape[:2]
#     resized_image = cv2.resize(image, (target_size, target_size))
#     scale_x = target_size / w
#     scale_y = target_size / h
#     resized_boxes = []
#     for box in boxes:
#         x1 = int(box[0] * scale_x)
#         y1 = int(box[1] * scale_y)
#         x2 = int(box[2] * scale_x)
#         y2 = int(box[3] * scale_y)
#         resized_boxes.append([x1, y1, x2, y2])
#     return resized_image, resized_boxes

# # Обработать каждый подкаталог
# with tqdm(total=len(os.listdir(src_folder)), desc="Processing directories") as pbar_dirs:
#     for task_folder in os.listdir(src_folder):
#         task_path = os.path.join(src_folder, task_folder)
#         if not os.path.isdir(task_path):
#             continue

#         # Создать соответствующий каталог в целевой папке
#         dst_task_path = os.path.join(dst_folder, task_folder)
#         os.makedirs(dst_task_path, exist_ok=True)

#         # Копировать изображения
#         images_src_path = os.path.join(task_path, 'images')
#         images_dst_path = os.path.join(dst_task_path, 'images')
#         os.makedirs(images_dst_path, exist_ok=True)

#         # Обработать JSON-файл аннотаций
#         annotations_path = os.path.join(task_path, 'annotations', 'instances_default.json')
#         with open(annotations_path, 'r') as f:
#             coco_data = json.load(f)

#         annotations = coco_data['annotations']
#         images = coco_data['images']
#         categories = class_id_to_name  # Используем словарь для перевода имен классов

#         jsonl_data = []

#         for image_info in images:
#             image_id = image_info['id']
#             image_filename = image_info['file_name']
#             width = image_info['width']
#             height = image_info['height']

#             image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]

#             suffix = ''
#             boxes = []
#             for ann in image_annotations:
#                 if ann['category_id'] not in categories:
#                     continue  # Пропустить аннотации с недействительными category_id
                
#                 category_name = categories[ann['category_id']]
#                 segmentation = ann['segmentation']

#                 # Преобразовать сегментации в маску и найти bounding boxes
#                 combined_mask = process_mask(ann, height, width)

#                 contours, _ = cv2.findContours(
#                     combined_mask.astype(np.uint8),
#                     cv2.RETR_TREE,
#                     cv2.CHAIN_APPROX_SIMPLE,
#                 )
                
#                 if not contours:
#                     continue # Пропустить, если контуры не найдены
                
#                 for contour in contours:
#                     x, y, w, h = cv2.boundingRect(contour)
#                     x2 = x + w
#                     y2 = y + h
#                     # Нормализовать и масштабировать bounding box
#                     box = [
#                         int(x * 1000 / width),
#                         int(y * 1000 / height),
#                         int(x2 * 1000 / width),
#                         int(y2 * 1000 / height)
#                     ]
#                     suffix += f"{category_name}<loc_{box[0]}><loc_{box[1]}><loc_{box[2]}><loc_{box[3]}>"
#                     boxes.append([x, y, x2, y2])

#             # Загрузить изображение
#             image_path = os.path.join(images_src_path, image_filename)
#             image = cv2.imread(image_path)
#             if image is None:
#                 continue

#             # Изменить размер изображения и bounding boxes
#             resized_image, resized_boxes = resize_image_and_boxes(image, boxes, 1024)

#             # Сохранить измененное изображение
#             dst_image_path = os.path.join(images_dst_path, image_filename)
#             cv2.imwrite(dst_image_path, resized_image)

#             # Обновить suffix с учетом новых размеров bounding boxes
#             suffix = ''
#             # тут ошибка
#             # print("len image_annotations", len(image_annotations)) 9
#             # print("len resized_boxes", len(resized_boxes)) 11
#             # print("annotations_path", annotations_path) sinusite_json_data/task_sinusite_data_29_11_23_1_st_sin_labeling/annotations/instances_default.json
#             # print("image_id", image_id) 68
#             for i, box in enumerate(resized_boxes):
#                 if i >= len(image_annotations):
#                     print(f"Skipping box {i} as it exceeds image_annotations length")
#                     break
#                 if image_annotations[i]['category_id'] not in categories:
#                     continue  # Пропустить аннотации с недействительными category_id
                
#                 category_name = categories[image_annotations[i]['category_id']]
#                 suffix += f"{category_name}<loc_{box[0]}><loc_{box[1]}><loc_{box[2]}><loc_{box[3]}>"

#             jsonl_data.append({
#                 "image": image_filename,
#                 "prefix": "<OD>",
#                 "suffix": suffix
#             })

#         # Записать результаты в JSONL файл
#         jsonl_file_path = os.path.join(dst_task_path, 'annotations.jsonl')
#         with open(jsonl_file_path, 'w') as f:
#             for entry in jsonl_data:
#                 json.dump(entry, f)
#                 f.write('\n')

#         pbar_dirs.update(1)








# тут как будто дубликаты

# # Указать пути к папкам
# src_folder = 'sinusite_json_data'
# dst_folder = 'sinusite_jsonl'

# # Создать целевую папку, если она не существует
# os.makedirs(dst_folder, exist_ok=True)

# # Словарь для перевода id в имя класса на английском языке
# class_id_to_name = {
#     1: "Right maxillary sinus (outer contour)",
#     2: "Left maxillary sinus (outer contour)",
#     3: "Left frontal sinus (outer contour)",
#     4: "Right frontal sinus (outer contour)",
#     5: "Right maxillary sinus (inner void boundary)",
#     6: "Left maxillary sinus (inner void boundary)",
#     7: "Left frontal sinus (inner void boundary)",
#     8: "Right frontal sinus (inner void boundary)",
#     9: "Reduction of pneumatization of paranasal sinuses",
#     10: "Horizontal fluid-air level",
#     11: "Absence of pneumatization of paranasal sinuses",
#     12: "Other pathology",
#     13: "Inscription"
# }

# # Функция для извлечения ограничивающих рамок из маски
# def process_mask(ann, image_height, image_width):
#     rles = maskUtils.frPyObjects(ann["segmentation"], image_height, image_width)
    
#     # Создаем пустую маску для текущей аннотации
#     combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)

#     # Если rles это не список, делаем его списком
#     if not isinstance(rles, list):
#         rles = [rles]

#     for rle in rles:
#         mask = maskUtils.decode(rle)

#         if len(mask.shape) == 3:
#             mask = np.max(mask, axis=2)

#         # Добавляем текущую маску к общей маске
#         combined_mask = np.maximum(combined_mask, mask)

#     return combined_mask

# # Функция для изменения размера изображения и корректировки bounding boxes
# def resize_image_and_boxes(image, boxes, target_size):
#     h, w = image.shape[:2]
#     resized_image = cv2.resize(image, (target_size, target_size))
#     scale_x = target_size / w
#     scale_y = target_size / h
#     resized_boxes = []
#     for box in boxes:
#         x1 = int(box[0] * scale_x)
#         y1 = int(box[1] * scale_y)
#         x2 = int(box[2] * scale_x)
#         y2 = int(box[3] * scale_y)
#         resized_boxes.append([x1, y1, x2, y2])
#     return resized_image, resized_boxes

# # Обработать каждый подкаталог
# with tqdm(total=len(os.listdir(src_folder)), desc="Processing directories") as pbar_dirs:
#     for task_folder in os.listdir(src_folder):
#         task_path = os.path.join(src_folder, task_folder)
#         if not os.path.isdir(task_path):
#             continue

#         # Создать соответствующий каталог в целевой папке
#         dst_task_path = os.path.join(dst_folder, task_folder)
#         os.makedirs(dst_task_path, exist_ok=True)

#         # Копировать изображения
#         images_src_path = os.path.join(task_path, 'images')
#         images_dst_path = os.path.join(dst_task_path, 'images')
#         os.makedirs(images_dst_path, exist_ok=True)

#         # Обработать JSON-файл аннотаций
#         annotations_path = os.path.join(task_path, 'annotations', 'instances_default.json')
#         with open(annotations_path, 'r') as f:
#             coco_data = json.load(f)

#         annotations = coco_data['annotations']
#         images = coco_data['images']
#         categories = class_id_to_name  # Используем словарь для перевода имен классов

#         jsonl_data = []

#         for image_info in images:
#             image_id = image_info['id']
#             image_filename = image_info['file_name']
#             width = image_info['width']
#             height = image_info['height']

#             image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]

#             suffix = ''
#             all_boxes = []
#             all_categories = []
#             for ann in image_annotations:
#                 if ann['category_id'] not in categories:
#                     continue  # Пропустить аннотации с недействительными category_id
                
#                 category_name = categories[ann['category_id']]
#                 segmentation = ann['segmentation']

#                 # Преобразовать сегментации в маску и найти bounding boxes
#                 combined_mask = process_mask(ann, height, width)

#                 contours, _ = cv2.findContours(
#                     combined_mask.astype(np.uint8),
#                     cv2.RETR_TREE,
#                     cv2.CHAIN_APPROX_SIMPLE,
#                 )
                
#                 if not contours:
#                     continue  # Пропустить, если контуры не найдены
                
#                 for contour in contours:
#                     x, y, w, h = cv2.boundingRect(contour)
#                     x2 = x + w
#                     y2 = y + h
#                     # Добавить bounding box в список
#                     all_boxes.append([x, y, x2, y2])
#                     all_categories.append(category_name)

#             # Загрузить изображение
#             image_path = os.path.join(images_src_path, image_filename)
#             image = cv2.imread(image_path)
#             if image is None:
#                 continue

#             # Изменить размер изображения и bounding boxes
#             resized_image, resized_boxes = resize_image_and_boxes(image, all_boxes, 1024)

#             # Сохранить измененное изображение
#             dst_image_path = os.path.join(images_dst_path, image_filename)
#             cv2.imwrite(dst_image_path, resized_image)

#             # Обновить suffix с учетом новых размеров bounding boxes
#             suffix = ''
#             for i, box in enumerate(resized_boxes):
#                 if i >= len(all_categories):
#                     print(f"Skipping box {i} as it exceeds categories length")
#                     break
#                 category_name = all_categories[i]
#                 suffix += f"{category_name}<loc_{box[0]}><loc_{box[1]}><loc_{box[2]}><loc_{box[3]}>"

#             jsonl_data.append({
#                 "image": image_filename,
#                 "prefix": "<OD>",
#                 "suffix": suffix
#             })

#         # Записать результаты в JSONL файл
#         jsonl_file_path = os.path.join(dst_task_path, 'annotations.jsonl')
#         with open(jsonl_file_path, 'w') as f:
#             for entry in jsonl_data:
#                 json.dump(entry, f)
#                 f.write('\n')

#         pbar_dirs.update(1)






# Указать пути к папкам
src_folder = 'sinusite_json_data'
dst_folder = 'sinusite_jsonl'

# Создать целевую папку, если она не существует
os.makedirs(dst_folder, exist_ok=True)

# Словарь для перевода id в имя класса на английском языке
class_id_to_name = {
    1: "Right maxillary sinus (outer contour)",
    2: "Left maxillary sinus (outer contour)",
    3: "Left frontal sinus (outer contour)",
    4: "Right frontal sinus (outer contour)",
    5: "Right maxillary sinus (inner void boundary)",
    6: "Left maxillary sinus (inner void boundary)",
    7: "Left frontal sinus (inner void boundary)",
    8: "Right frontal sinus (inner void boundary)",
    9: "Reduction of pneumatization of paranasal sinuses",
    10: "Horizontal fluid-air level",
    11: "Absence of pneumatization of paranasal sinuses",
    12: "Other pathology",
    13: "Inscription"
}

# Функция для извлечения ограничивающих рамок из маски
def process_mask(ann, image_height, image_width):
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

    return combined_mask

# Функция для изменения размера изображения и корректировки bounding boxes
def resize_image_and_boxes(image, boxes, target_size):
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

# Обработать каждый подкаталог
with tqdm(total=len(os.listdir(src_folder)), desc="Processing directories") as pbar_dirs:
    for task_folder in os.listdir(src_folder):
        task_path = os.path.join(src_folder, task_folder)
        if not os.path.isdir(task_path):
            continue

        # Создать соответствующий каталог в целевой папке
        dst_task_path = os.path.join(dst_folder, task_folder)
        os.makedirs(dst_task_path, exist_ok=True)

        # Копировать изображения
        images_src_path = os.path.join(task_path, 'images')
        images_dst_path = os.path.join(dst_task_path, 'images')
        os.makedirs(images_dst_path, exist_ok=True)

        # Обработать JSON-файл аннотаций
        annotations_path = os.path.join(task_path, 'annotations', 'instances_default.json')
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)

        annotations = coco_data['annotations']
        images = coco_data['images']
        categories = class_id_to_name  # Используем словарь для перевода имен классов

        jsonl_data = []

        for image_info in images:
            image_id = image_info['id']
            image_filename = image_info['file_name']
            width = image_info['width']
            height = image_info['height']

            image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]

            suffix = ''
            all_boxes = []
            all_categories = []
            for ann in image_annotations:
                if ann['category_id'] not in categories:
                    continue  # Пропустить аннотации с недействительными category_id
                
                category_name = categories[ann['category_id']]
                segmentation = ann['segmentation']

                # Преобразовать сегментации в маску и найти bounding boxes
                combined_mask = process_mask(ann, height, width)

                contours, _ = cv2.findContours(
                    combined_mask.astype(np.uint8),
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                
                if not contours:
                    continue  # Пропустить, если контуры не найдены
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    x2 = x + w
                    y2 = y + h
                    # Добавить bounding box в список
                    all_boxes.append([x, y, x2, y2])
                    all_categories.append(category_name)

            # Загрузить изображение
            image_path = os.path.join(images_src_path, image_filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            # Изменить размер изображения и bounding boxes
            resized_image, resized_boxes = resize_image_and_boxes(image, all_boxes, 1024)

            # Удалить дубликаты боксов
            unique_boxes = set()
            unique_resized_boxes = []
            unique_categories = []
            for i, box in enumerate(resized_boxes):
                if tuple(box) not in unique_boxes:
                    unique_boxes.add(tuple(box))
                    unique_resized_boxes.append(box)
                    unique_categories.append(all_categories[i])
            resized_boxes = unique_resized_boxes
            all_categories = unique_categories

            # Сохранить измененное изображение
            dst_image_path = os.path.join(images_dst_path, image_filename)
            cv2.imwrite(dst_image_path, resized_image)

            # Обновить suffix с учетом новых размеров bounding boxes
            suffix = ''
            for i, box in enumerate(resized_boxes):
                if i >= len(all_categories):
                    print(f"Skipping box {i} as it exceeds categories length")
                    break
                category_name = all_categories[i]
                suffix += f"{category_name}<loc_{box[0]}><loc_{box[1]}><loc_{box[2]}><loc_{box[3]}>"

            jsonl_data.append({
                "image": image_filename,
                "prefix": "<OD>",
                "suffix": suffix
            })

        # Записать результаты в JSONL файл
        jsonl_file_path = os.path.join(dst_task_path, 'annotations.jsonl')
        with open(jsonl_file_path, 'w') as f:
            for entry in jsonl_data:
                json.dump(entry, f)
                f.write('\n')

        pbar_dirs.update(1)