# тут много фоток сразу модель прогонит и сохранит 


import os
import json
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import matplotlib.pyplot as plt  
import matplotlib.patches as patches  

# Устройство для работы с моделью
DEVICE = torch.device("cuda:2")

# Загрузка модели и процессора
model_checkpoint = "/home/imran-nasyrov/model_checkpoints/1.10/epoch_789"
model = AutoModelForCausalLM.from_pretrained(model_checkpoint, trust_remote_code=True).eval().to(DEVICE)
processor = AutoProcessor.from_pretrained(model_checkpoint, trust_remote_code=True)

# Пути к файлам
jsonl_path = "/home/imran-nasyrov/cvat_sinonyms/task_task_13_oct_23_pat_fut_1c-2024_02_26_15_44_35-coco 1.0.zip/annotations.jsonl"
images_folder = "/home/imran-nasyrov/cvat_sinonyms/task_task_13_oct_23_pat_fut_1c-2024_02_26_15_44_35-coco 1.0.zip/images"
save_folder = "/home/imran-nasyrov/test_infer_florence"  # Папка для сохранения изображений

# Убедимся, что папка для сохранения существует
os.makedirs(save_folder, exist_ok=True)

# Считываем первые 300 строк из файла jsonl
def read_jsonl(file_path, num_lines=300):
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            data.append(json.loads(line.strip()))
    return data

# Функция для генерации результата с моделью
def run_example(task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].to(DEVICE),
      pixel_values=inputs["pixel_values"].to(DEVICE),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer, image

# Функция для рисования bounding box
def plot_bbox(image, data, save_path, text_input):
    # Create a figure and axes  
    fig, ax = plt.subplots()  

    # Display the image  
    ax.imshow(image)  
    
    image_width, image_height = image.size

    # Plot each bounding box  
    for bbox, label in zip(data['bboxes'], data['labels']):  
        # Unpack the bounding box coordinates  
        x1, y1, x2, y2 = bbox  
        # Create a Rectangle patch  
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')  
        # Add the rectangle to the Axes  
        ax.add_patch(rect)  
        # Annotate the label  
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))  

    plt.text(image_width - 10, image_height - 10, text_input, color='white', fontsize=12, ha='right', va='bottom',
             bbox=dict(facecolor='black', alpha=0.7))
    
    # Remove the axis ticks and labels  
    ax.axis('off')  

    # Show the plot  
    # save_path = save_folder
    
    if text_input != "":
        fig.savefig(f"{save_path}", bbox_inches='')
        # fig.savefig(f"{save_path}/{text_input}.jpg", bbox_inches='')
    else:
        fig.savefig(f"{save_path}/img.jpg", bbox_inches='')

# Основная функция для обработки строк jsonl
def process_jsonl_data(jsonl_data):
    for entry in tqdm(jsonl_data):
        image_name = entry["image"]
        prefix = entry["prefix"]
        text_input = prefix.split("CAPTION_TO_PHRASE_GROUNDING ")[1]  # Извлекаем подсказку
        image_path = os.path.join(images_folder, image_name)

        # Загружаем изображение
        image = Image.open(image_path)

        # Запускаем модель
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        results, modified_image = run_example(task_prompt, image, text_input=text_input)

        # Сохраняем результат
        save_image_name = f"{os.path.splitext(image_name)[0]}_{text_input}.jpg"
        save_path = os.path.join(save_folder, save_image_name)
        print("save_path", save_path)

        # Рисуем и сохраняем bounding box
        plot_bbox(modified_image, results['<CAPTION_TO_PHRASE_GROUNDING>'], save_path, text_input)
        print(f"Saved: {save_image_name}")

# Чтение данных из jsonl
jsonl_data = read_jsonl(jsonl_path)

# Обработка данных и сохранение изображений
process_jsonl_data(jsonl_data)
