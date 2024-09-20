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
DEVICE = torch.device("cuda:0")

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


root_directory_path = "/home/imran-nasyrov/cvat_mini"
# root_directory_path = "/home/imran-nasyrov/cvat_phrase"
# root_directory_path = "/home/imran-nasyrov/cvat_jsonl"
# root_directory_path = "/home/imran-nasyrov/sinusite_jsonl"

# вот так надо!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# train_subdirectories, val_subdirectories = split_directories(root_directory_path)

# а это для mini
# train_subdirectories = ["/home/imran-nasyrov/cvat_mini/task_task_17_jule_23_pathology_anatomy_sinus_num2-2023_09_12_22_50_06-coco 1.0"]
# val_subdirectories = ["/home/imran-nasyrov/cvat_mini/task_task_17_jule_23_pathology_anatomy_sinus_num3-2023_09_06_00_11_03-coco 1.0"]


# это чисто легкие
train_subdirectories = [
    "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_1c-2024_02_26_15_44_35-coco 1.0",
    "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_2с-2024_02_16_02_27_35-coco 1.0",
    "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_3c-2024_04_24_19_39_16-coco 1.0",
    "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_4c-2024_03_29_13_08_59-coco 1.0",
    "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_6с-2024_04_02_10_23_13-coco 1.0",
    "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_7с-2024_03_15_17_38_41-coco 1.0",
    "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_8с-2024_02_27_18_08_45-coco 1.0",
    "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_9с-2023_11_14_19_54_06-coco 1.0"
    "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_10с-2024_05_12_10_15_51-coco 1.0",
    "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_11с-2023_11_16_09_13_16-coco 1.0",
    "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_12с-2024_03_01_18_54_17-coco 1.0",
]
val_subdirectories = [
    "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_13с-2023_11_19_18_58_39-coco 1.0",
    "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_5с-2024_04_09_10_42_06-coco 1.0",
]

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


# def draw_annotations(image, prefix, suffix):
#     # Разбиваем суффикс на класс и координаты
#     parts = suffix.split('<loc_')
#     class_name = parts[0].rstrip('<>')

#     height, width, _ = image.shape

#     for i in range(1, len(parts), 4):
#         if i + 3 < len(parts):
#             try:
#                 # Преобразование координат обратно к оригинальным размерам
#                 x1 = int(parts[i].split('>')[0]) * width // 1000
#                 y1 = int(parts[i + 1].split('>')[0]) * height // 1000
#                 x2 = int(parts[i + 2].split('>')[0]) * width // 1000
#                 y2 = int(parts[i + 3].split('>')[0]) * height // 1000

#                 # Нарисовать прямоугольник (bounding box)
#                 cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

#                 # Подписать класс
#                 cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#             except (IndexError, ValueError) as e:
#                 print(f"Error processing bounding box: {e}")

#     # Наносим текст префикса и суффикса на изображение
#     cv2.putText(image, f"Prefix: {prefix}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#     cv2.putText(image, f"Suffix: {class_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

#     return image


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
        try:
            model.train()
            train_loss = 0
            for inputs, answers in tqdm(
                train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"
            ):
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
                
                # # print("inputs", inputs)
                # # print("answers", answers)
                
                # logits = outputs.logits  # (batch_size, seq_len, vocab_size)
                # # print("logits", logits)
                # # print("logits shape", logits.shape)
                # # Reshape logits to (batch_size * seq_len, vocab_size)
                # logits = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
                # # print("logits 2", logits)
                # # print("logits 2 shape", logits.shape)
                # # Reshape labels to (batch_size * seq_len)
                # labels = labels.view(-1)  # (batch_size * seq_len)
                # # print("logits 3", logits)
                # # print("logits 3 shape", logits.shape)
                # # Calculate the loss using custom cross entropy
                
                # loss = custom_cross_entropy(logits, labels)
                
                
                
                # вот так еще попробую
                logits = outputs.logits

                # Применяем кастомный лосс
                logits = logits.view(-1, logits.size(-1))  # Преобразуем логиты в (batch_size * seq_len, vocab_size)
                labels = labels.view(-1)  # Преобразуем метки в (batch_size * seq_len)
                
                loss = custom_cross_entropy(logits, labels)
                
                ###########################

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                train_loss += loss.item()
                # n+=1
                # print("train_loss", train_loss / n)
                

                

            avg_train_loss = train_loss / len(train_loader)
            print(f"Average Training Loss: {avg_train_loss}")

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, answers in tqdm(
                    val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"
                ):
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
                    
                    # logits = outputs.logits  # (batch_size, seq_len, vocab_size)
                    # logits = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
                    # labels = labels.view(-1)  # (batch_size * seq_len)
                    # loss = custom_cross_entropy(logits, labels)
                    
                    # вот так попробую
                    logits = outputs.logits

                    # Применяем кастомный лосс
                    logits = logits.view(-1, logits.size(-1))  # Преобразуем логиты в (batch_size * seq_len, vocab_size)
                    labels = labels.view(-1)  # Преобразуем метки в (batch_size * seq_len)
                    
                    loss = custom_cross_entropy(logits, labels)
                    ####################
                    
                    val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                print(f"Average Validation Loss: {avg_val_loss}")

                # render_inference_results(peft_model, val_loader.dataset, 6)

        
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Loss/validation", avg_val_loss, epoch)

            output_dir = f"./model_checkpoints/{experiment_name}/epoch_{epoch+1}"
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

        except:
            pass
        
    writer.close()


EPOCHS = 1500
LR = 5e-6
experiment_name = "1.7"

train_model(
    experiment_name,
    train_loader,
    val_loader,
    peft_model,
    processor,
    epochs=EPOCHS,
    lr=LR,
)



# import os

# # Указываем директорию, куда будем сохранять изображения и соответствующие данные
# output_dir = "florence_dataloader"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Итерируемся по train_loader или val_loader без обучения
# for i, (inputs, answers) in enumerate(train_loader):
#     # Получаем изображения и префиксы/суффиксы
#     images = inputs["pixel_values"]  # изображения в тензорах
#     prefixes = inputs["input_ids"]  # префиксы в тензорах
#     suffixes = answers  # суффиксы, уже в текстовом формате

#     # Обрабатываем каждое изображение в батче
#     for j in range(images.size(0)):
#         # Преобразуем изображение обратно в PIL формат для сохранения
#         image = images[j].cpu().detach().numpy()
#         image = np.transpose(image, (1, 2, 0))  # Приводим размерности в порядок
#         image = (image * 255).astype(np.uint8)  # Возвращаем из нормализации
#         image_pil = Image.fromarray(image)
        
#         # Создаем имя файла на основе префикса
#         prefix_text = processor.tokenizer.decode(prefixes[j].cpu().detach().numpy(), skip_special_tokens=True)
#         prefix_clean = prefix_text.replace(" ", "_").replace("/", "_")  # Убираем недопустимые символы
        
#         # Определяем путь сохранения изображения
#         image_save_path = os.path.join(output_dir, f"{prefix_clean}_{i}_{j}.jpg")
        
#         # Сохраняем изображение
#         image_pil.save(image_save_path)
        
#         # Сохраняем суффикс в текстовый файл
#         suffix_save_path = os.path.join(output_dir, f"{prefix_clean}_{i}_{j}_suffix.txt")
#         with open(suffix_save_path, "w") as f:
#             f.write(suffixes[j])
    
#     # Останавливаемся на небольшом количестве итераций для проверки
#     if i > 5:  # Если нужно, измените это значение для большего количества данных
#         break

# print(f"Сохранено {i * BATCH_SIZE} изображений и их суффиксов в директорию {output_dir}")





# import os
# import cv2
# import numpy as np
# import torch

# # Создаем директорию для сохранения изображений, если она не существует
# output_image_dir = "florence_dataloader"
# os.makedirs(output_image_dir, exist_ok=True)

# def save_images_with_annotations(batch_images, batch_prefixes, batch_suffixes, epoch):
#     """
#     Сохраняет изображения с аннотациями (префиксами и суффиксами).
    
#     :param batch_images: Тензор изображений (batch_size, 3, height, width)
#     :param batch_prefixes: Список префиксов
#     :param batch_suffixes: Список суффиксов
#     :param epoch: Номер текущей эпохи
#     """
#     for i in range(len(batch_images)):
#         prefix = batch_prefixes[i]
#         suffix = batch_suffixes[i]
        
#         # Извлекаем изображение из тензора и преобразуем его в формат OpenCV
#         image = batch_images[i].cpu().numpy().transpose(1, 2, 0)
#         image = (image * 255).astype(np.uint8)
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Преобразуем в формат BGR для OpenCV
        
#         # Обрабатываем суффикс и рисуем аннотации
#         annotated_image = draw_annotations(image, prefix, suffix)
        
#         # Сохраняем изображение
#         cv2.imwrite(f"{output_image_dir}/epoch_{epoch}_image_{i}.jpg", annotated_image)

# def draw_annotations(image, prefix, suffix):
#     """
#     Добавляет текстовые аннотации и рисует bounding box на изображении.
    
#     :param image: Исходное изображение (numpy массив)
#     :param prefix: Префикс (текст)
#     :param suffix: Суффикс с координатами для bounding box
#     :return: Изображение с аннотациями
#     """
#     # Рисуем текст префикса
#     cv2.putText(image, prefix, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#     # Обработка и разделение суффикса на класс и координаты
#     parts = suffix.split('<loc_')
#     class_name = parts[0].rstrip('<>')

#     height, width, _ = image.shape
#     for i in range(1, len(parts), 4):
#         if i + 3 < len(parts):
#             try:
#                 x1 = int(parts[i].split('>')[0]) * width // 1000
#                 y1 = int(parts[i + 1].split('>')[0]) * height // 1000
#                 x2 = int(parts[i + 2].split('>')[0]) * width // 1000
#                 y2 = int(parts[i + 3].split('>')[0]) * height // 1000

#                 # Рисуем прямоугольник и текст класса
#                 cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#             except (IndexError, ValueError) as e:
#                 print(f"Error processing bounding box: {e}")

#     return image

# # Пример вызова функции после загрузки батча данных
# def process_batch(batch, epoch):
#     inputs, answers = batch
#     batch_images = inputs["pixel_values"]
#     batch_prefixes = inputs["input_ids"]  # Если это токены, нужно декодировать их в текст
#     batch_prefixes = [processor.decode(prefix) for prefix in batch_prefixes]  # Преобразуем в текст

#     # Сохраняем изображения с аннотациями
#     save_images_with_annotations(batch_images, batch_prefixes, answers, epoch)


# # Вызываем функцию для каждого батча в DataLoader
# for epoch in range(EPOCHS):
#     for i, batch in enumerate(train_loader):
#         # Обрабатываем и сохраняем изображения
#         process_batch(batch, epoch)
        
#         # (Опционально) Если вы хотите сохранить только несколько батчей для проверки
#         if i >= 5:  # Сохраняем только первые 5 батчей
#             break