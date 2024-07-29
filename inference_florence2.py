import os
import torch
from torch.utils.data import DataLoader
import cv2
from PIL import Image
from transformers import get_scheduler
from transformers.optimization import AdamW
from transformers.peft import LoraConfig, get_peft_model
from tqdm import tqdm
import random

# Определите функцию для отрисовки предсказаний
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
        annotations = answer.split('<loc_')
        class_name = annotations[0].rstrip('<>')

        # Загрузите изображение
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width, _ = image_cv.shape

        for j in range(1, len(annotations), 4):
            if j + 3 < len(annotations):
                try:
                    x1 = int(annotations[j].split('>')[0]) * width // 1000
                    y1 = int(annotations[j + 1].split('>')[0]) * height // 1000
                    x2 = int(annotations[j + 2].split('>')[0]) * width // 1000
                    y2 = int(annotations[j + 3].split('>')[0]) * height // 1000

                    # Нарисуйте прямоугольник и подпись
                    cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image_cv, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except (IndexError, ValueError) as e:
                    print(f"Error processing bounding box: {e}")

        # Сохраните изображение
        output_path = os.path.join(output_directory, f"result_{i}.jpg")
        cv2.imwrite(output_path, image_cv)

# Загрузите обученную модель и процессор
model_checkpoint = "./model_checkpoints/epoch_10"
model = AutoModelForVision2Seq.from_pretrained(model_checkpoint)
processor = AutoProcessor.from_pretrained(model_checkpoint)

# Пример вызова функции render_inference_results
output_directory = "./inference_results"
render_inference_results(model, val_dataset, count=6, output_directory=output_directory)
