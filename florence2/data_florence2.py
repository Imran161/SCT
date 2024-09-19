from pycocotools import mask as maskUtils
import json
import os

import cv2
import numpy as np
from tqdm import tqdm
import sys

# # это для синуситов я сам перевел классы

# # Указать пути к папкам
# src_folder = "sinusite_json_data"
# dst_folder = "test_sinusite_jsonl"

# # src_folder = "cvat_unzip"
# # dst_folder = "cvat_jsonl"

# # Создать целевую папку, если она не существует
# os.makedirs(dst_folder, exist_ok=True)

# # это фиксированные классы
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
#     13: "Inscription",
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
#         # вот так было, но тут на 1000 не умножается
#         # x1 = int(box[0] * scale_x)
#         # y1 = int(box[1] * scale_y)
#         # x2 = int(box[2] * scale_x)
#         # y2 = int(box[3] * scale_y)
#         # resized_boxes.append([x1, y1, x2, y2])

#         x1 = int(box[0] * scale_x)
#         y1 = int(box[1] * scale_y)
#         x2 = int(box[2] * scale_x)
#         y2 = int(box[3] * scale_y)

#         # Масштабирование до диапазона [0, 999]
#         scaled_x1 = int(x1 * 1000 / target_size)
#         scaled_y1 = int(y1 * 1000 / target_size)
#         scaled_x2 = int(x2 * 1000 / target_size)
#         scaled_y2 = int(y2 * 1000 / target_size)

#         resized_boxes.append([scaled_x1, scaled_y1, scaled_x2, scaled_y2])
#     return resized_image, resized_boxes


# # Обработать каждый подкаталог
# with tqdm(
#     total=len(os.listdir(src_folder)), desc="Processing directories"
# ) as pbar_dirs:
#     for task_folder in os.listdir(src_folder):
#         task_path = os.path.join(src_folder, task_folder)
#         if not os.path.isdir(task_path):
#             continue

#         # Создать соответствующий каталог в целевой папке
#         dst_task_path = os.path.join(dst_folder, task_folder)
#         os.makedirs(dst_task_path, exist_ok=True)

#         # Обработать JSON-файл аннотаций
#         annotations_path = os.path.join(
#             task_path, "annotations", "instances_default.json"
#         )
#         with open(annotations_path, "r") as f:
#             coco_data = json.load(f)

#         annotations = coco_data["annotations"]
#         images = coco_data["images"]

#         # categories = class_id_to_name  # Используем словарь для перевода имен классов
#         # Создание словаря для перевода id в имя класса из JSON файла
#         categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

#         jsonl_data = []

#         for image_info in images:
#             image_id = image_info["id"]
#             image_filename = image_info["file_name"]
#             width = image_info["width"]
#             height = image_info["height"]

#             image_annotations = [
#                 ann for ann in annotations if ann["image_id"] == image_id
#             ]

#             suffix = ""
#             all_boxes = []
#             all_categories = []
#             for ann in image_annotations:
#                 if ann["category_id"] not in categories:
#                     continue  # Пропустить аннотации с недействительными category_id

#                 category_name = categories[ann["category_id"]]
#                 segmentation = ann["segmentation"]

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
#             image_path = os.path.join(task_path, "images", image_filename)
#             image = cv2.imread(image_path)
#             if image is None:
#                 continue

#             # Изменить размер изображения и bounding boxes
#             resized_image, resized_boxes = resize_image_and_boxes(
#                 image, all_boxes, 1024
#             )

#             # print("resized_image", resized_image.shape)
#             # print("combined_mask", combined_mask.shape)
#             # break
            
#             # Удалить дубликаты боксов
#             unique_boxes = set()
#             unique_resized_boxes = []
#             unique_categories = []
#             for i, box in enumerate(resized_boxes):
#                 if tuple(box) not in unique_boxes:
#                     unique_boxes.add(tuple(box))
#                     unique_resized_boxes.append(box)
#                     unique_categories.append(all_categories[i])
#             resized_boxes = unique_resized_boxes
#             all_categories = unique_categories

#             # Сформировать путь для сохранения изображения в целевой папке
#             dst_image_path = os.path.join(dst_task_path, "images", image_filename)

#             # Создать необходимые подкаталоги
#             os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)

#             # Сохранить измененное изображение
#             cv2.imwrite(dst_image_path, resized_image)

#             # Обновить suffix с учетом новых размеров bounding boxes
#             suffix = ""
#             for i, box in enumerate(resized_boxes):
#                 if i >= len(all_categories):
#                     print(f"Skipping box {i} as it exceeds categories length")
#                     break
#                 category_name = all_categories[i]
#                 suffix += f"{category_name}<loc_{box[0]}><loc_{box[1]}><loc_{box[2]}><loc_{box[3]}>"

#             jsonl_data.append(
#                 {
#                     "image": image_filename,
#                     "prefix": "<CAPTION_TO_PHRASE_GROUNDING>",  # <OD> было
#                     "suffix": suffix,
#                 }
#             )

#         # Записать результаты в JSONL файл
#         jsonl_file_path = os.path.join(dst_task_path, "annotations.jsonl")
#         with open(jsonl_file_path, "w") as f:
#             for entry in jsonl_data:
#                 json.dump(entry, f)
#                 f.write("\n")

#         pbar_dirs.update(1)


# это переводит на английский язык классы

import os
import zipfile
import json
import urllib.parse
from googletrans import Translator
from tqdm import tqdm

# Путь к исходной папке с архивами
source_dir = "/home/imran-nasyrov/cvat/"

# Путь к папке, куда будут распаковываться архивы
destination_dir = "/home/imran-nasyrov/cvat_unzip/"

# Файл для записи уже обработанных папок
processed_dirs_file = "/home/imran-nasyrov/SCT/florence2/processed_dirs.txt"

# Инициализация переводчика
translator = Translator()
translator.raise_exception = True  # Исправляем с 'raise_Exception'

# Проверяем, существует ли целевая директория, если нет — создаем её
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Загружаем список уже обработанных папок
if os.path.exists(processed_dirs_file):
    with open(processed_dirs_file, "r") as f:
        processed_dirs = set(line.strip() for line in f)
else:
    processed_dirs = set()


# Функция для декодирования и перевода категорий
def decode_and_translate_categories(categories):
    translated_categories = []
    russian_names = []
    for category in categories:
        # Декодирование имени категории с UTF-8
        decoded_name = urllib.parse.unquote(category["name"])
        print("decoded_name", decoded_name)
        russian_names.append(decoded_name)

        # Проверка, что категория уже на английском (ASCII)
        if all(ord(char) < 128 for char in decoded_name):
            translated_categories.append(category)
            continue

        # Перевод на английский язык
        translated_name = translator.translate(decoded_name, src="ru", dest="en").text

        # Обновление имени категории
        category["name"] = translated_name
        translated_categories.append(category)

    print("russian_names", russian_names)
    return translated_categories


# Проходимся по всем файлам в исходной папке
with tqdm(total=len(os.listdir(source_dir)), desc="Processing directories") as pbar_dirs:
    for filename in os.listdir(source_dir):
        if filename.endswith(".zip"):
            # print("filename", filename)
            if "task_task_13_oct_23_pat_fut_1c-2024_02_26_15_44_35-coco 1.0.zip" in filename:
                # Определяем имя папки, которая будет создана для распаковки
                folder_name = filename[:-4]  # Убираем '.zip' из имени файла
                # sys.exit()
                
                # Пропускаем папки, которые уже были обработаны
                if folder_name in processed_dirs:
                    print(f"Папка {folder_name} уже обработана, пропускаем.")
                    pbar_dirs.update(1)
                    continue

                # Определяем полный путь к архиву
                file_path = os.path.join(source_dir, filename)

                folder_path = os.path.join(destination_dir, folder_name)

                # Создаем папку для распакованных файлов
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                # Распаковываем архив в созданную папку
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(folder_path)

                print(f"Архив {filename} успешно распакован в {folder_path}")

                # Путь к файлу instances_default.json
                json_file_path = os.path.join(
                    folder_path, "annotations", "instances_default.json"
                )

                # Проверяем, существует ли файл
                if os.path.exists(json_file_path):
                    # Открываем и читаем JSON файл
                    with open(json_file_path, "r", encoding="utf-8") as json_file:
                        data = json.load(json_file)

                    # Декодируем и переводим категории
                    try:
                        data["categories"] = decode_and_translate_categories(data["categories"])
###########################################
                        # # Сохраняем изменения обратно в JSON файл
                        # with open(json_file_path, "w", encoding="utf-8") as json_file:
                        #     json.dump(data, json_file, ensure_ascii=False)

                        # print(f"Файл {json_file_path} обновлен.")

                        # # Добавляем папку в список обработанных
                        # with open(processed_dirs_file, "a") as f:
                        #     f.write(folder_name + "\n")
                        # processed_dirs.add(folder_name)
############################################
                    except Exception as e:
                        print(f"Ошибка при обработке файла {json_file_path}: {e}")
                        break  # Прерываем выполнение при ошибке

                pbar_dirs.update(1)

print("Процесс завершен.")
