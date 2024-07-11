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
from PIL import Image, ImageDraw, ImageFont
import random
import matplotlib.pyplot as plt
from transformers import AutoProcessor, get_scheduler
from torch.optim import AdamW


set_seed(64)

CHECKPOINT = "microsoft/Florence-2-base-ft"
REVISION = "refs/pr/6"
DEVICE = torch.device("cuda:0")

model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, trust_remote_code=True, revision=REVISION
).to(DEVICE)
processor = AutoProcessor.from_pretrained(
    CHECKPOINT, trust_remote_code=True, revision=REVISION
)

processor.image_processor.size = {"height": 1024, "width": 1024}
processor.image_processor.do_rescale = False

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


def draw_polygons(image, prediction, fill_mask=False, save_path=None):
    """
    Draws segmentation masks with polygons on an image.

    Parameters:
    - image_path: Path to the image file.
    - prediction: Dictionary containing 'polygons' and 'labels' keys.
                  'polygons' is a list of lists, each containing vertices of a polygon.
                  'labels' is a list of labels corresponding to each polygon.
    - fill_mask: Boolean indicating whether to fill the polygons with color.
    """
    # Load the image

    draw = ImageDraw.Draw(image)


    # Set up scale factor if needed (use 1 if not scaling)
    scale = 1

    # Iterate over polygons and labels
    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = "red"
        fill_color = "red" if fill_mask else None

        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue

            _polygon = (_polygon * scale).reshape(-1).tolist()

            # Draw the polygon
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)

            # Draw the label text
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    image.save(f"{save_path}/img.jpg")
    print(f"Saved image with polygons drawn to: {save_path}")


def save_raw_images(images, output_path, start_index):
    """
    Save raw images from a batch for verification.
    
    Parameters:
    - images: Tensor of images.
    - output_path: Directory to save the images.
    - start_index: Starting index for image filenames.
    """
    for i, image_tensor in enumerate(images):
        image = image_tensor.permute(1, 2, 0).cpu().numpy()
        # print("image shape", image.shape)
        # print("image_tensor shape", image_tensor.shape)
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        
        image.save(f"{output_path}/raw_image_{start_index + i}.png")
        print(f"Saved raw image to: {output_path}/raw_image_{start_index + i}.png")

        # do_rescale=False

def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-8)
    return normalized_tensor
            
def render_inference_results(model, data_loader, output_path: str, max_count: int):
    model.eval()
    count = 0
    with torch.no_grad():
        for batch in data_loader:
            if count >= max_count:
                break

            inputs, targets = batch
            
            

            # проверю сами картинки
            # img = min_max_normalize(inputs["pixel_values"])
            # print("img shape", img.shape)
            # save_raw_images(img, output_path, count)
            
            # print("targets[masks].shape", len(targets["masks"]))
            # save_raw_images(targets["masks"], output_path, count)
            
            generated_ids = model.generate(
                input_ids=inputs["input_ids"].to(DEVICE),
                pixel_values=inputs["pixel_values"].to(DEVICE),
                max_new_tokens=1024,
                num_beams=3,
            )

            # print(type(inputs["pixel_values"]))
            # print(inputs["pixel_values"].size())

            generated_text = processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )  # [0]
            answers = [
                processor.post_process_generation(
                    text,
                    task="<REFERRING_EXPRESSION_SEGMENTATION>",
                    image_size=(img.size(-2), img.size(-1)),
                )
                for text, img in zip(generated_text, inputs["pixel_values"])
            ]

            print("answers[0]", answers[0])
            
            for idx, (image_tensor, answer) in enumerate(zip(inputs["pixel_values"], answers)):
                # Преобразуем тензор изображения в PIL Image
                print("image_tensor shape", image_tensor.shape)
                image = image_tensor.permute(1, 2, 0).cpu().numpy()
                image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)

                mask = answer["<REFERRING_EXPRESSION_SEGMENTATION>"]
                # print("mask", mask)
                draw_polygons(
                        image,
                        mask, #answer["<REFERRING_EXPRESSION_SEGMENTATION>"],
                        fill_mask=True,
                        save_path=f"{output_path}/annotated_image_{count + idx}.png"
                    )

                # save_images_with_masks(
                #     inputs["pixel_values"],
                #     masks,
                #     output_path,
                #     count,
                # )
            count += len(inputs["pixel_values"])


# Путь для сохранения аннотированных изображений
output_path = "test_infer_florence"

# render_inference_results(peft_model, val_loader, output_path, 4)

def polygons_to_mask(polygons, image_size):
    mask = np.zeros(image_size, dtype=np.uint8)
    for polygon in polygons:
        pts = np.array(polygon, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
    return torch.tensor(mask, dtype=torch.float32)#, requires_grad=True)


def resize_masks(masks, target_size):
    resized_masks = []
    for mask in masks:
        resized_mask = cv2.resize(mask.detach().cpu().numpy(), target_size, interpolation=cv2.INTER_NEAREST)
        resized_masks.append(torch.tensor(resized_mask, dtype=torch.float32).unsqueeze(0))
    return torch.cat(resized_masks, dim=0)



def train_model(train_loader, val_loader, model, processor, epochs=10, lr=1e-6):
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n = len(train_loader)
        desc = f"Epoch {epoch + 1}/{epochs}"

        with tqdm(total=n, desc=desc, unit="batch") as pbar:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
            # for inputs, targets in tqdm(train_loader, desc=desc):
                input_ids = inputs["input_ids"].to(DEVICE)
                pixel_values = inputs["pixel_values"].to(DEVICE)
                target_masks = torch.stack(targets["masks"]).to(DEVICE)
                
                # Преобразование целевых масок в однослойные и изменение их размера
                target_masks = target_masks[:, 0, :, :]  # Используем первый канал
                target_masks = resize_masks(target_masks, (768, 768)).to(DEVICE)

                optimizer.zero_grad()

                generated_ids = model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=1024,
                    num_beams=3,
                )
                generated_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )
                answers = [
                    processor.post_process_generation(
                        text,
                        task="<REFERRING_EXPRESSION_SEGMENTATION>",
                        image_size=(img.size(-2), img.size(-1)),
                    )
                    for text, img in zip(generated_text, pixel_values)
                ]
                
                pred_masks = [
                    polygons_to_mask(answer["<REFERRING_EXPRESSION_SEGMENTATION>"]["polygons"], (img.size(-2), img.size(-1)))
                    for answer, img in zip(answers, pixel_values)
                ]
                
                pred_masks = resize_masks(pred_masks, (768, 768)).to(DEVICE).requires_grad_()
                # print("pred_masks", pred_masks)
                loss = criterion(pred_masks, target_masks)
                
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                train_loss += loss.item()
                
                pbar.set_postfix(loss=train_loss / (batch_idx + 1))
                pbar.update(1)

        avg_train_loss = train_loss / n
        print(f"Average Training Loss: {avg_train_loss}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                input_ids = inputs["input_ids"].to(DEVICE)
                pixel_values = inputs["pixel_values"].to(DEVICE)
                target_masks = torch.stack(targets["masks"]).to(DEVICE)

                # Преобразование целевых масок в однослойные и изменение их размера
                target_masks = target_masks[:, 0, :, :]  # Используем первый канал
                target_masks = resize_masks(target_masks, (768, 768)).to(DEVICE)

                generated_ids = model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=1024,
                    num_beams=3,
                )
                generated_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )
                answers = [
                    processor.post_process_generation(
                        text,
                        task="<REFERRING_EXPRESSION_SEGMENTATION>",
                        image_size=(img.size(-2), img.size(-1)),
                    )
                    for text, img in zip(generated_text, pixel_values)
                ]
                
                pred_masks = [
                    polygons_to_mask(answer["<REFERRING_EXPRESSION_SEGMENTATION>"]["polygons"], (img.size(-2), img.size(-1)))
                    for answer, img in zip(answers, pixel_values)
                ]
                
                # Изменение размера предсказанных масок
                pred_masks = resize_masks(pred_masks, (768, 768)).to(DEVICE)

                loss = criterion(pred_masks, target_masks)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Average Validation Loss: {avg_val_loss}")

            render_inference_results(model, val_loader, f"output_path_epoch_{epoch+1}", 6)

        output_dir = f"./model_checkpoints/epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        

criterion = nn.CrossEntropyLoss()
train_model(train_loader, val_loader, model, processor, epochs=10, lr=1e-6)
