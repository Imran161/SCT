from typing import List, Dict, Any, Tuple
import random
import os
import json
import torch

import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, AutoModelForCausalLM, AutoProcessor, get_scheduler
import torch.nn.functional as F
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from PIL import Image


CHECKPOINT = "microsoft/Florence-2-base-ft"
REVISION = "refs/pr/6"
DEVICE = torch.device("cuda:2")

model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, trust_remote_code=True, revision=REVISION
).to(DEVICE)
processor = AutoProcessor.from_pretrained(
    CHECKPOINT, trust_remote_code=True, revision=REVISION
)


# @title Define `JSONLDataset` class


class JSONLDataset:
    def __init__(self, subdirectories: List[str]):
        self.subdirectories = subdirectories
        self.entries = self._load_entries()

    def _load_entries(self) -> List[Dict[str, Any]]:
        entries = []
        for subdir in self.subdirectories:
            jsonl_file_path = os.path.join(subdir, "annotations.jsonl")
            image_directory_path = os.path.join(subdir, "images")
            if os.path.exists(jsonl_file_path) and os.path.exists(image_directory_path):
                with open(jsonl_file_path, "r") as file:
                    for line in file:
                        data = json.loads(line)
                        data["image_directory_path"] = image_directory_path
                        entries.append(data)

        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")

        entry = self.entries[idx]
        image_path = os.path.join(entry["image_directory_path"], entry["image"])
       
        try:
            # было так 
            # image = Image.open(image_path).convert("RGB")
            
            # Загрузка черно-белого изображения с использованием OpenCV
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Image file {image_path} not found.")

            # Конвертация черно-белого изображения в RGB формат
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        
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
        
        prefix = data["prefix"]
        suffix = data["suffix"]
        return prefix, suffix, image


# @title Function to split directories into train and validation


def split_directories(root_directory_path: str, train_val_ratio: float = 0.9):
    subdirectories = [
        os.path.join(root_directory_path, d)
        for d in os.listdir(root_directory_path)
        if os.path.isdir(os.path.join(root_directory_path, d))
    ]

    # в task_sinusite_data_29_11_23_1_st_sin_labeling , 2,3 в этих трех файлах не скопировались фотки, там исправлять надо
    # wrong_list = ["/home/imran-nasyrov/sinusite_jsonl/task_sinusite_data_29_11_23_1_st_sin_labeling",
    #             "/home/imran-nasyrov/sinusite_jsonl/task_sinusite_data_29_11_23_2_st_sin_labeling",
    #             "/home/imran-nasyrov/sinusite_jsonl/task_sinusite_data_29_11_23_3_st_sin_labeling"
    # ]

    # subdirectories = [subdirs for subdirs in subdirectories if subdirs not in wrong_list]
    # print("subdirectories", subdirectories)
    random.shuffle(subdirectories)

    num_train = int(train_val_ratio * len(subdirectories))
    train_subdirectories = subdirectories[:num_train]
    val_subdirectories = subdirectories[num_train:]

    print("train_subdirectories", train_subdirectories)
    print("val_subdirectories", val_subdirectories)
    
    return train_subdirectories, val_subdirectories


BATCH_SIZE = 6
NUM_WORKERS = 0


def collate_fn(batch):
    questions, answers, images = zip(*batch)
    # print("questions", questions)
    # print("answers", answers)
    # # print("images", len(images))
    
    # for img in images:
    #     print("img.shape", img.sum())
        
    inputs = processor(
        text=list(questions), images=list(images), return_tensors="pt", padding=True
    ).to(DEVICE)
    return inputs, answers


root_directory_path = "/home/imran-nasyrov/cvat_sinonyms"
# root_directory_path = "/home/imran-nasyrov/cvat_mini"
# root_directory_path = "/home/imran-nasyrov/cvat_phrase"
# root_directory_path = "/home/imran-nasyrov/cvat_jsonl"
# root_directory_path = "/home/imran-nasyrov/sinusite_jsonl"

# вот так надо!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
train_subdirectories, val_subdirectories = split_directories(root_directory_path)

# а это для mini
# train_subdirectories = ["/home/imran-nasyrov/cvat_mini/task_task_17_jule_23_pathology_anatomy_sinus_num2-2023_09_12_22_50_06-coco 1.0"]
# val_subdirectories = ["/home/imran-nasyrov/cvat_mini/task_task_17_jule_23_pathology_anatomy_sinus_num3-2023_09_06_00_11_03-coco 1.0"]


# это чисто легкие
# train_subdirectories = [
#     "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_1c-2024_02_26_15_44_35-coco 1.0",
#     "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_2с-2024_02_16_02_27_35-coco 1.0",
#     "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_3c-2024_04_24_19_39_16-coco 1.0",
#     "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_4c-2024_03_29_13_08_59-coco 1.0",
#     "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_6с-2024_04_02_10_23_13-coco 1.0",
#     "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_7с-2024_03_15_17_38_41-coco 1.0",
#     "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_8с-2024_02_27_18_08_45-coco 1.0",
#     "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_9с-2023_11_14_19_54_06-coco 1.0"
#     "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_10с-2024_05_12_10_15_51-coco 1.0",
#     "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_11с-2023_11_16_09_13_16-coco 1.0",
#     "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_12с-2024_03_01_18_54_17-coco 1.0",
# ]
# val_subdirectories = [
#     "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_13с-2023_11_19_18_58_39-coco 1.0",
#     "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_5с-2024_04_09_10_42_06-coco 1.0",
# ]

train_dataset = DetectionDataset(subdirectories=train_subdirectories)
val_dataset = DetectionDataset(subdirectories=val_subdirectories)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,    
    shuffle=False,
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS
)

print("len train_loader", len(train_loader))
print("val val_loader", len(val_loader))

# в task_sinusite_data_29_11_23_1_st_sin_labeling , 2,3 в этих трех файлах не скопировались фотки, там исправлять надо


# @title Setup LoRA Florence-2 model

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

torch.cuda.empty_cache()


# def custom_cross_entropy(logits, labels):
#     """
#     Логиты: (batch_size, num_classes)
#     Лейблы: (batch_size,)
#     """
#     # Шаг 1: Применяем softmax к логитам
#     probs = torch.softmax(logits, dim=-1)  # (batch_size, num_classes)
#     # print("probs", probs)
#     # print("probs shape", probs.shape)
    
#     # Шаг 2: Берем вероятность правильного класса
#     true_class_probs = probs[torch.arange(len(labels)), labels]  # (batch_size,)
    
#     # Шаг 3: Вычисляем отрицательный логарифм вероятности правильного класса
#     loss = -torch.log(true_class_probs)  # (batch_size,)
    
#     # Шаг 4: Усредняем потери по всем примерам
#     return loss.mean()


def custom_cross_entropy(logits, labels):
    # Примените вашу логику для кастомного лосса
    # Например, это может быть обычная кросс-энтропия с модификациями
    # Логиты должны быть нормализованы перед применением лосса
    loss = F.cross_entropy(logits, labels)
    return loss


SMOOTH = 1e-8

class GlobalFocusLoss:
    def __init__(self, mode="ML"):
        self.mode = mode
        self.global_loss_sum = torch.tensor(0.0, dtype=torch.double)
        self.global_loss_numel = torch.tensor(0.0, dtype=torch.double)

    def forward(self, input: torch.Tensor, target: torch.Tensor, train_mode=True):
        # print("input before", input)
        
        input = torch.softmax(input, dim=-1)
        
        # print("input", input)
        # print("target", target)
        
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=input.size(-1)).float()

        # проверить numel
        
        if self.mode == "ML":
            loss_bce = -(
                target_one_hot * torch.log(input + SMOOTH)
                + (1 - target_one_hot) * torch.log(1 - input + SMOOTH)
            )
        elif self.mode == "MC":
            loged_target = torch.log(input + SMOOTH)
            loss_bce = -target_one_hot * loged_target
            
            # print("input", input)
            # print("loged_target", loged_target)
            # print("loss_bce", loss_bce)
            
            # loss_bce = суммировать по измерению токенов вот тут
            # loss_bce = loss_bce.sum(dim=-1)

            
            # метрики добавить 
        
        # if self.mode == "ML":
        #     loss_bce = -(
        #         target * torch.log(input + SMOOTH)
        #         + (1 - target) * torch.log(1 - input + SMOOTH)
        #     )

        # elif self.mode == "MC":
        #     loged_target = torch.log(input + SMOOTH)
        #     loss_bce = -target * loged_target

        if train_mode:
            self.global_loss_sum += loss_bce.sum().item()
            self.global_loss_numel += loss_bce.numel()
            
            # Установка предела для global_loss_sum и global_loss_numel
            max_value = 1e8  # или любое другое значение, которое вы считаете приемлемым
            self.global_loss_sum = torch.clamp(self.global_loss_sum, max=max_value)
            self.global_loss_numel = torch.clamp(self.global_loss_numel, max=max_value)


            pt = torch.exp(loss_bce - self.global_loss_sum / self.global_loss_numel)
            loss = loss_bce * pt
            loss_mean = torch.mean(loss)

        else:
            loss_mean = torch.mean(loss_bce)

        return loss_mean

    def reset_global_loss(self):
        """Сбросить накопленные значения потерь."""
        self.global_loss_sum = torch.tensor(0.0, dtype=torch.double)
        self.global_loss_numel = torch.tensor(0.0, dtype=torch.double)


class BCELoss:
    @staticmethod
    def forward(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # тут добавил softmax
        input = torch.softmax(input, dim=-1)

        target_one_hot = torch.nn.functional.one_hot(target.to(torch.int64), num_classes=input.size(-1)).float()

        loss = -target_one_hot * torch.log(input + SMOOTH) - (1 - target_one_hot) * torch.log(
            1 - input + SMOOTH
        )
        return loss
    
    
class FocalLoss:
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor = None,
        reduction: str = "mean",
        normalized: bool = False,
        reduced_threshold=None,
        eps: float = 1e-4,
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.normalized = normalized
        self.reduced_threshold = reduced_threshold
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        size = target.shape
        target = target.type(input.type())

        loss_ce = BCELoss.forward(input, target)

        pt = torch.exp(-loss_ce)

        if self.reduced_threshold is None:
            focal_term = (1.0 - pt).pow(self.gamma)
        else:
            focal_term = ((1.0 - pt) / self.reduced_threshold).pow(self.gamma)
            focal_term[pt < self.reduced_threshold] = 1

        loss_focal = focal_term * loss_ce

        if self.alpha is not None:
            for i in range(size[0]):
                for j in range(size[1]):
                    weight_matrix = (
                        (target[i, j]) * self.alpha[0][j]
                        + (1 - target[i, j]) * self.alpha[1][j]
                    )
                    loss_focal[i, j] = loss_focal[i, j] * weight_matrix

        if self.reduction == "mean":
            loss_focal = loss_focal.mean()
        elif self.reduction == "sum":
            loss_focal = loss_focal.sum()
        elif self.reduction == "batchwise_mean":
            loss_focal = loss_focal.sum(0)

        return loss_focal



# criterion = FocalLoss()
criterion = GlobalFocusLoss(mode="MC") # MC надо было сделать 


def train_model(
    experiment_name, train_loader, val_loader, model, processor, epochs=10, lr=1e-6
):
    writer = SummaryWriter(log_dir=f"runs_florence2/{experiment_name}_logs")

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # n=0
    
    output_image_dir = "florence_dataloader"
    os.makedirs(output_image_dir, exist_ok=True)
    
    for epoch in range(epochs):
        # try:
        model.train()
        train_loss = 0
        
        with tqdm(
            total=len(train_loader),
            desc=f"Training Epoch {epoch + 1}/{epochs}",
            unit="batch",
        ) as pbar:
        
            for batch_idx, (inputs, answers) in enumerate(val_loader):
                
        # for inputs, answers in tqdm(
        #     train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"
        # ):
            # max_lenght = 1024
            
                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                
                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True, # "max_length", # True,
                    return_token_type_ids=False,
                    # max_length=1024
                    truncation=True #можно еще попробовать
                ).input_ids.to(DEVICE)
                
                # print("input_ids", input_ids)
                # # print("pixel_values", pixel_values)
                # print("labels", labels)
                    
                outputs = model(
                    input_ids=input_ids, pixel_values=pixel_values, labels=labels
                )
                # print("outputs", outputs)
                # break
                

                # Декодируем labels и logits обратно в текст для вывода
                decoded_labels = processor.batch_decode(labels, skip_special_tokens=False)
                decoded_outputs = processor.batch_decode(
                    torch.argmax(outputs.logits, dim=-1), skip_special_tokens=False
                )
                
                # Печатаем вход и выход модели
                # print(f"True Labels: {decoded_labels}")
                # print(f"Model Outputs: {decoded_outputs}")
                #########################################
                # было так
                # print("labels shape", labels.shape)
                # print("labels", labels)
                # print("outputs shape", outputs["logits"].shape)
                # print("outputs", outputs["logits"])
                
                # loss = outputs.loss
                ##########################################
                
                # тут мой лосс

                # logits = outputs.logits
                # # print("logits", logits) тут отрицательные числа

                # # Применяем кастомный лосс
                # logits = logits.view(-1, logits.size(-1))  # Преобразуем логиты в (batch_size * seq_len, vocab_size)
                # labels = labels.view(-1)  # Преобразуем метки в (batch_size * seq_len)
                
                # # BCE кастомный
                # # loss = custom_cross_entropy(logits, labels)
                
                #########################################
                # focus loss 
                
                logits = outputs.logits
                # print("logits", logits) #тут отрицательные числа

                # Применяем кастомный лосс
                logits = logits.view(-1, logits.size(-1))  # Преобразуем логиты в (batch_size * seq_len, vocab_size)
                labels = labels.view(-1)  # Преобразуем метки в (batch_size * seq_len)
                
                # print("logits", logits.shape)
                # print("labels", labels.shape)
                loss = criterion.forward(logits, labels, train_mode=True)
                ###########################
                # focal loss
                
                # logits = outputs.logits
                # logits = logits.view(-1, logits.size(-1))  # Преобразуем логиты в (batch_size * seq_len, vocab_size)
                # labels = labels.view(-1)  # Преобразуем метки в (batch_size * seq_len)
                
                # loss = criterion.forward(logits, labels)
                ###########################

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                train_loss += loss.item()
                # n+=1
                # print("train_loss", train_loss)
                
                avg_loss = train_loss / (batch_idx + 1)
                # Обновляем tqdm с текущим средним лоссом
                pbar.set_postfix(loss=avg_loss)
                pbar.update(1)

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            with tqdm(
                total=len(val_loader),
                desc=f"Validation Epoch {epoch + 1}/{epochs}",
                unit="batch",
            ) as pbar:
                
                for batch_idx, (inputs, answers) in enumerate(val_loader):
                
            # for inputs, answers in tqdm(
            #     val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"
            # ):
                    input_ids = inputs["input_ids"]
                    pixel_values = inputs["pixel_values"]
                    labels = processor.tokenizer(
                        text=answers,
                        return_tensors="pt",
                        padding=True,
                        return_token_type_ids=False,
                        truncation=True
                    ).input_ids.to(DEVICE)

                    

                    
                    # max_lenght = 1024
                    # labels = labels[:,:max_lenght]
                    
                    outputs = model(
                        input_ids=input_ids, pixel_values=pixel_values, labels=labels
                    )
                    
                    # Декодируем labels и logits обратно в текст для вывода
                    decoded_labels = processor.batch_decode(labels, skip_special_tokens=False)
                    decoded_outputs = processor.batch_decode(
                        torch.argmax(outputs.logits, dim=-1), skip_special_tokens=False
                    )
                    
                    # Печатаем вход и выход модели
                    # print(f"True Labels: {decoded_labels}")
                    # print(f"Model Outputs: {decoded_outputs}")
                
                    # было так
                    ##############################
                    # loss = outputs.loss
                    #############################
                    # мой лосс 

                    # logits = outputs.logits

                    # # Применяем кастомный лосс
                    # logits = logits.view(-1, logits.size(-1))  # Преобразуем логиты в (batch_size * seq_len, vocab_size)
                    # labels = labels.view(-1)  # Преобразуем метки в (batch_size * seq_len)
                    
                    # # BCE кастомный
                    # # loss = custom_cross_entropy(logits, labels)
                    
                    ###############################
                    # focus
                    
                    logits = outputs.logits
                    logits = logits.view(-1, logits.size(-1))  # Преобразуем логиты в (batch_size * seq_len, vocab_size)
                    labels = labels.view(-1)  # Преобразуем метки в (batch_size * seq_len)
                    
                    criterion.reset_global_loss() 
                    loss = criterion.forward(logits, labels, train_mode=False)
                    ####################
                    # focal
                    
                    # logits = outputs.logits
                    # logits = logits.view(-1, logits.size(-1))  # Преобразуем логиты в (batch_size * seq_len, vocab_size)
                    # labels = labels.view(-1)  # Преобразуем метки в (batch_size * seq_len)
                    
                    # loss = criterion.forward(logits, labels)
                    ####################
                    
                    val_loss += loss.item()
                    
                    avg_val_loss = val_loss / (batch_idx + 1)

                    # Обновляем tqdm с текущим средним валидационным лоссом
                    pbar.set_postfix(loss=avg_val_loss)
                    pbar.update(1)

            avg_val_loss = val_loss / len(val_loader)
            print(f"Average Validation Loss: {avg_val_loss}")

                # render_inference_results(peft_model, val_loader.dataset, 6)

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/validation", avg_val_loss, epoch)

        output_dir = f"./model_checkpoints/{experiment_name}/epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)

        # except:
        #     pass
        
    writer.close()


EPOCHS = 1500
LR = 5e-6
experiment_name = "1.10"

train_model(
    experiment_name,
    train_loader,
    val_loader,
    peft_model,
    processor,
    epochs=EPOCHS,
    lr=LR,
)


