import os
import io
import json
from PIL import Image
from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset
import itertools
import os
import warnings

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForImageSegmentation,
    AutoProcessor,
    SegformerConfig,
    SegformerFeatureExtractor,
    SegformerForSemanticSegmentation,
)

from coco_classes import sinusite_base_classes, sinusite_pat_classes_3
from coco_dataloaders import SINUSITE_COCODataLoader, FLORENCE_COCODataLoader
from metrics import DetectionMetrics
from sct_val import test_model
from transforms import SegTransform
from utils import ExperimentSetup, iou_metric, save_best_metrics_to_csv, set_seed
from peft import LoraConfig, get_peft_model
import supervision as sv
import base64
from PIL import Image
from IPython.display import display, HTML


set_seed(64)

CHECKPOINT = "microsoft/Florence-2-base-ft"
REVISION = "refs/pr/6"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, trust_remote_code=True, revision=REVISION
).to(DEVICE)
processor = AutoProcessor.from_pretrained(
    CHECKPOINT, trust_remote_code=True, revision=REVISION
)


BATCH_SIZE = 6
NUM_CLASSES = 2

params = {
    "json_file_path": "/home/imran-nasyrov/sinusite_json_data",
    "delete_list": [],
    "base_classes": sinusite_base_classes,
    "out_classes": sinusite_pat_classes_3,
    "dataloader": True,
    "resize": (1024, 1024),
    "recalculate": False,
    "delete_null": False,
}

coco_dataloader = FLORENCE_COCODataLoader(params)

(
    train_loader,
    val_loader,
    total_train,
    pixel_total_train,
    list_of_name_out_classes,
) = coco_dataloader.make_dataloaders(batch_size=BATCH_SIZE, train_val_ratio=0.8)


config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "linear",
        "Conv2d",
        "lm_head",
        "fc2",
    ],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
    bias="none",
    inference_mode=False,
    use_rslora=True,
    init_lora_weights="gaussian",
    revision=REVISION,
)

peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()


def save_images_with_masks(images, masks, output_path: str, start_index: int):
    for i, (image, mask) in enumerate(zip(images, masks)):
        image = image.squeeze().cpu().numpy()
        mask = mask.squeeze().cpu().numpy()

        # Преобразуем маску в формат, подходящий для OpenCV
        mask = (mask * 255).astype(np.uint8)

        # Объединяем изображение и маску
        color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        combined_image = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)

        # Сохраняем изображение с аннотациями
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        cv2.imwrite(
            f"{output_path}/annotated_image_{start_index + i}.png", combined_image
        )


def render_inference_results(model, data_loader, output_path: str, max_count: int):
    model.eval()
    count = 0
    with torch.no_grad():
        for batch in data_loader:
            if count >= max_count:
                break

            inputs, targets = batch  # to(DEVICE) надо наверное
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

            generated_text = processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )  # [0]
            answers = [
                processor.post_process_generation(
                    text,
                    task="<REFERRING_EXPRESSION_SEGMENTATION>",
                    image_size=img.size,
                )
                for text, img in zip(generated_text, inputs["pixel_values"])
            ]

            # Сохраняем изображения с масками
            save_images_with_masks(
                inputs["pixel_values"],
                [answer["masks"] for answer in answers],
                output_path,
                count,
            )
            count += len(inputs["pixel_values"])


# Путь для сохранения аннотированных изображений
output_path = "test_infer_florence"

render_inference_results(peft_model, val_loader, output_path, 4)
