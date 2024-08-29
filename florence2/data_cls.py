import json
import os

import cv2
import numpy as np
import torch
from pycocotools import mask as maskUtils
from tqdm import tqdm

from ..transforms import SegTransform


class DataProcessor:
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


# Использование класса для обработки данных
src_folder = "sinusite_json_data"
dst_folder = "test_sinusite_jsonl"

task_type = "OD"  # "OD" или "CAPTION_TO_PHRASE_GROUNDING" или "REFERRING_EXPRESSION_SEGMENTATION"
augment = True

processor = DataProcessor(src_folder, dst_folder, task_type, augment=augment)
processor.process_directory()
