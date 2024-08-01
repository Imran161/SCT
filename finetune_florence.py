import io
import os
import re
import json
import torch
import html
import base64
import itertools

import numpy as np
import cv2
import supervision as sv

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoProcessor,
    get_scheduler
)
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Generator
from peft import LoraConfig, get_peft_model
from PIL import Image


CHECKPOINT = "microsoft/Florence-2-base-ft"
REVISION = 'refs/pr/6'
DEVICE = torch.device("cuda:2")

model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION).to(DEVICE)
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION)



import os
import json
import random
from typing import List, Dict, Any, Tuple
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# @title Define `JSONLDataset` class

class JSONLDataset:
    def __init__(self, subdirectories: List[str]):
        self.subdirectories = subdirectories
        self.entries = self._load_entries()

    def _load_entries(self) -> List[Dict[str, Any]]:
        entries = []
        for subdir in self.subdirectories:
            jsonl_file_path = os.path.join(subdir, 'annotations.jsonl')
            image_directory_path = os.path.join(subdir, 'images')
            if os.path.exists(jsonl_file_path) and os.path.exists(image_directory_path):
                with open(jsonl_file_path, 'r') as file:
                    for line in file:
                        data = json.loads(line)
                        data['image_directory_path'] = image_directory_path
                        entries.append(data)
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")

        entry = self.entries[idx]
        image_path = os.path.join(entry['image_directory_path'], entry['image'])
        try:
            image = Image.open(image_path).convert("RGB")
            return (image, entry)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file {image_path} not found.")

class DetectionDataset(Dataset):
    def __init__(self, subdirectories: List[str]):
        self.dataset = JSONLDataset(subdirectories)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, data = self.dataset[idx]
        prefix = data['prefix']
        suffix = data['suffix']
        return prefix, suffix, image

# @title Function to split directories into train and validation

def split_directories(root_directory_path: str, train_val_ratio: float = 0.9):
    subdirectories = [os.path.join(root_directory_path, d) for d in os.listdir(root_directory_path) if os.path.isdir(os.path.join(root_directory_path, d))]
    
    # в task_sinusite_data_29_11_23_1_st_sin_labeling , 2,3 в этих трех файлах не скопировались фотки, там исправлять надо
    wrong_list = ["/home/imran-nasyrov/sinusite_jsonl/task_sinusite_data_29_11_23_1_st_sin_labeling", 
                "/home/imran-nasyrov/sinusite_jsonl/task_sinusite_data_29_11_23_2_st_sin_labeling", 
                "/home/imran-nasyrov/sinusite_jsonl/task_sinusite_data_29_11_23_3_st_sin_labeling"
    ]
    
    subdirectories = [subdirs for subdirs in subdirectories if subdirs not in wrong_list]
    # print("subdirectories", subdirectories)
    random.shuffle(subdirectories)

    num_train = int(train_val_ratio * len(subdirectories))
    train_subdirectories = subdirectories[:num_train]
    val_subdirectories = subdirectories[num_train:]

    return train_subdirectories, val_subdirectories

# @title Initiate `DetectionsDataset` and `DataLoader` for train and validation subsets

BATCH_SIZE = 24
NUM_WORKERS = 0

def collate_fn(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(DEVICE)
    return inputs, answers

root_directory_path = "/home/imran-nasyrov/sinusite_jsonl"
train_subdirectories, val_subdirectories = split_directories(root_directory_path)

train_dataset = DetectionDataset(subdirectories=train_subdirectories)
val_dataset = DetectionDataset(subdirectories=val_subdirectories)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS)

print("len train_loader", len(train_loader))
print("val val_loader", len(val_loader))

# в task_sinusite_data_29_11_23_1_st_sin_labeling , 2,3 в этих трех файлах не скопировались фотки, там исправлять надо



# @title Setup LoRA Florence-2 model

config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
    bias="none",
    inference_mode=False,
    use_rslora=True,
    init_lora_weights="gaussian",
    revision=REVISION
)

peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()

torch.cuda.empty_cache()


# @title Define train loop

def train_model(experiment_name, train_loader, val_loader, model, processor, epochs=10, lr=1e-6):
    writer = SummaryWriter(log_dir=f"runs_florence2/{experiment_name}_logs")
    
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # render_inference_results(peft_model, val_loader.dataset, 6)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, answers in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):

            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False
            ).input_ids.to(DEVICE)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward(), optimizer.step(), lr_scheduler.step(), optimizer.zero_grad()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, answers in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):

                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False
                ).input_ids.to(DEVICE)

                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Average Validation Loss: {avg_val_loss}")

            # render_inference_results(peft_model, val_loader.dataset, 6)

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/validation", avg_val_loss, epoch)
        
        output_dir = f"./model_checkpoints/epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        
    writer.close()    

EPOCHS = 120
LR = 5e-6
experiment_name = "1.1"

# train_model(experiment_name, train_loader, val_loader, peft_model, processor, epochs=EPOCHS, lr=LR)



# Веса загрузить не могу

def render_inference_results(model, dataset: DetectionDataset, count: int, output_directory: str):
    os.makedirs(output_directory, exist_ok=True)
    count = min(count, len(dataset))
    for i in range(count):
        image, data = dataset.dataset[i]
        prefix = data['prefix']
        inputs = processor(text=prefix, images=image, return_tensors="pt").to(DEVICE)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        answer = processor.post_process_generation(generated_text, task='<OD>', image_size=image.size)

        # Получите аннотации из ответа
        print("answer", answer)
        bboxes = answer['<OD>']['bboxes']
        labels = answer['<OD>']['labels']

        # Преобразуйте изображение в формат OpenCV
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Нарисуйте bounding boxes и подписи
        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Сохраните изображение
        output_path = os.path.join(output_directory, f"result_{i}.jpg")
        cv2.imwrite(output_path, image_cv)


# Загрузите обученную модель и процессор
model_checkpoint = "/home/imran-nasyrov/model_checkpoints/epoch_120/"

model = AutoModelForCausalLM.from_pretrained(model_checkpoint, trust_remote_code=True).to(DEVICE)
processor = AutoProcessor.from_pretrained(model_checkpoint, trust_remote_code=True)

output_directory = "./inference_results"
render_inference_results(model, val_dataset, count=6, output_directory=output_directory)


