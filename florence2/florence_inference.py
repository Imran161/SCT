# тут много фоток сразу модель прогонит и сохранит 


import os
import json
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import matplotlib.pyplot as plt  
import matplotlib.patches as patches  

import onnxruntime as ort
import numpy as np
import torch
from transformers import AutoProcessor
from PIL import Image

from torchvision import transforms


# Устройство для работы с моделью
DEVICE = torch.device("cuda:1")

# Загрузка модели и процессора
model_checkpoint = "/home/imran-nasyrov/model_checkpoints/1.11/epoch_31"
model = AutoModelForCausalLM.from_pretrained(model_checkpoint, trust_remote_code=True).eval().to(DEVICE)
processor = AutoProcessor.from_pretrained(model_checkpoint, trust_remote_code=True)

# Пути к файлам
jsonl_path = "/home/imran-nasyrov/cvat_sinonyms/task_task_13_oct_23_pat_fut_1c-2024_02_26_15_44_35-coco 1.0.zip/annotations.jsonl"
images_folder = "/home/imran-nasyrov/cvat_sinonyms/task_task_13_oct_23_pat_fut_1c-2024_02_26_15_44_35-coco 1.0.zip/images"
save_folder = "/home/imran-nasyrov/test_infer_florence_1_11"  # Папка для сохранения изображений

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


#######################################
# Сохранение процессора

# processor_save_path = "/home/imran-nasyrov/processor"
# processor.save_pretrained(processor_save_path)
# print(f"Процессор сохранен в: {processor_save_path}")

# # Позже вы можете загрузить его так:
# processor = AutoProcessor.from_pretrained(processor_save_path, trust_remote_code=True)
# print("Процессор загружен.")

#######################################


#######################################
# Сохранение токенайзера в файл

# tokenizer_save_path = "/home/imran-nasyrov/tokenizer"  # Укажите путь к папке для сохранения токенайзера
# processor.tokenizer.save_pretrained(tokenizer_save_path)
# print(f"Токенайзер сохранен в: {tokenizer_save_path}")


# from transformers import AutoTokenizer

# # Загрузка токенайзера из сохранённого файла
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)
# print("Токенайзер загружен.")

###########################################

# Функция для генерации результата с моделью
# def run_example(task_prompt, image, text_input=None):
#     if text_input is None:
#         prompt = task_prompt
#     else:
#         prompt = task_prompt + text_input

#     inputs = processor(text=prompt, images=image, return_tensors="pt")
#     generated_ids = model.generate(
#       input_ids=inputs["input_ids"].to(DEVICE),
#       pixel_values=inputs["pixel_values"].to(DEVICE),
#       max_new_tokens=1024,
#       early_stopping=False,
#       do_sample=False,
#       num_beams=3,
#     )
    
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
#     parsed_answer = processor.post_process_generation(
#         generated_text,
#         task=task_prompt,
#         image_size=(image.width, image.height)
#     )

#     return parsed_answer, image


# Путь для сохранения тензоров
input_ids_path = "input_ids.pt"
pixel_values_path = "pixel_values.pt"
decoder_input_ids_path = "decoder_input_ids.pt"


# Сохранение тензоров в файлы
def save_tensors(inputs):
    input_ids = inputs["input_ids"].to(DEVICE)
    pixel_values = inputs["pixel_values"].to(DEVICE)
    decoder_input_ids = torch.tensor([[0]]).to(DEVICE)
    
    # Сохранение input_ids
    torch.save(input_ids, input_ids_path)
    print(f"input_ids сохранены в файл {input_ids_path}")

    # Сохранение pixel_values
    torch.save(pixel_values, pixel_values_path)
    print(f"pixel_values сохранены в файл {pixel_values_path}")
    
    # Сохранение decoder_input_ids
    torch.save(decoder_input_ids, decoder_input_ids_path)
    print(f"decoder_input_ids сохранены в файл {decoder_input_ids_path}")


class ONNXModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ONNXModelWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, pixel_values, decoder_input_ids):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            decoder_input_ids = decoder_input_ids,
            return_dict=True
        )
        return outputs.logits  # Возвращаем только логиты


def preprocess_image(image_path, image_size=(768, 768)):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(image_size), 
        transforms.ToTensor(),         
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    tensor_image = transform(image)
    tensor_image = tensor_image.unsqueeze(0)
    
    return tensor_image



def run_example(task_prompt, image, text_input=None, save_model=False, save_inputs=False, image_tensor=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    pixel_values = inputs["pixel_values"].to(DEVICE)
    # print("pixel_values shape", pixel_values.shape)
    # print("pixel_values", pixel_values)
    print("input_ids shape", input_ids.shape)
    print("input_ids", input_ids)

    # Получаем start_token_id для декодера
    # start_token_id = model.config.decoder_start_token_id
    # if start_token_id is None:
    #     raise ValueError("Model does not have a decoder_start_token_id defined in its configuration.")
    decoder_input_ids = torch.tensor([[0]], device=DEVICE)

    # Экспорт модели
    if save_model:
        onnx_model = ONNXModelWrapper(model)
        onnx_save_path = "florence2_epoch_136.onnx"
        torch.onnx.export(
            onnx_model,
            (input_ids, pixel_values, decoder_input_ids),
            onnx_save_path,
            export_params=True,
            opset_version=14,
            input_names=["input_ids", "pixel_values", "decoder_input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "pixel_values": {0: "batch_size"},
                "decoder_input_ids": {0: "batch_size", 1: "decoder_sequence_length"},
                "logits": {0: "batch_size", 1: "decoder_sequence_length"}
            }
        )
        print(f"Модель сохранена в формате ONNX по пути: {onnx_save_path}")

    # Инференс с ONNX-моделью
    ort_session = ort.InferenceSession("florence2.onnx")
    input_ids_np = input_ids.cpu().numpy()
    pixel_values_np = pixel_values.cpu().numpy()
    
    image_tensor_np = image_tensor.cpu().numpy()

    # Реализация цикла генерации
    # generated_tokens = [start_token_id]
    generated_tokens = [0]
    max_length = 60  # Установите желаемую максимальную длину

    for step in range(max_length):
        decoder_input_ids = torch.tensor([generated_tokens], dtype=torch.long)
        decoder_input_ids_np = decoder_input_ids.cpu().numpy()

        outputs = ort_session.run(
            None,
            {
                "input_ids": input_ids_np,
                "pixel_values": pixel_values_np, # image_tensor_np для проверки моей предобработки
                "decoder_input_ids": decoder_input_ids_np
            }
        )
        logits = outputs[0]
        next_token_logits = logits[:, -1, :]
        next_token_id = np.argmax(next_token_logits, axis=-1)[0]

        generated_tokens.append(int(next_token_id))

        if next_token_id == model.config.eos_token_id:
            break
        
    print("generated_tokens", generated_tokens)
    
    # Декодирование сгенерированных токенов
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=False)
    print("Сгенерированный текст:", generated_text)

    # Постобработка и возврат результатов
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer, image




def my_run_example(task_prompt, image, text_input=None, save_model=False, save_inputs=False, image_tensor=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    if save_inputs:
        save_tensors(inputs)
        
        
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
    
        
    # print("generated_text", generated_text)
    # print("generated_ids", generated_ids)
    # print("generated_ids shape", generated_ids.shape)
    # print("parsed_answer", parsed_answer)

    # Сохранение модели в ONNX и TorchScript форматах (если включено)
    if save_model:
        # example_inputs = (inputs["input_ids"].to(DEVICE), inputs["pixel_values"].to(DEVICE))
        
        example_inputs = {
            "input_ids": inputs["input_ids"].to(DEVICE),
            "pixel_values": inputs["pixel_values"].to(DEVICE),
            "decoder_input_ids": torch.tensor([[0]]).to(DEVICE)  # Пример decoder_input_ids
        }
        
        torch_example_inputs = (
            inputs["input_ids"].to(DEVICE),
            inputs["pixel_values"].to(DEVICE),
            torch.tensor([[0]]).to(DEVICE)  # Пример decoder_input_ids
        )
        
        # Сохранение модели в формате ONNX
        # onnx_save_path = "florence2_1.onnx"
        # torch.onnx.export(
        #     model,
        #     example_inputs,
        #     onnx_save_path,
        #     export_params=True,
        #     opset_version=14,
        #     input_names=["input_ids", "pixel_values"],
        #     output_names=["logits"],
        #     dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"},
        #     "pixel_values": {0: "batch_size"},
        #     "logits": {0: "batch_size", 1: "sequence_length"}
        #     }
        # )
        # print(f"Модель сохранена в формате ONNX по пути: {onnx_save_path}")
        
        
############################################################################
        # второй способ
        
        # Создайте экземпляр обёрнутой модели
        onnx_model = ONNXModelWrapper(model)

        # Подготовьте пример входных данных
        input_ids = inputs["input_ids"].to(DEVICE)
        pixel_values = inputs["pixel_values"].to(DEVICE)
        # decoder_input_ids = torch.tensor([[0]]).to(DEVICE) 
        
        start_token_id = model.config.decoder_start_token_id
        if start_token_id is None:
            raise ValueError("Model does not have a decoder_start_token_id defined in its configuration.")

        decoder_input_ids = torch.tensor([[start_token_id]], device=DEVICE)

        
        # Экспорт модели
        onnx_save_path = "florence2.onnx"
        torch.onnx.export(
            onnx_model,
            (input_ids, pixel_values, decoder_input_ids),
            onnx_save_path,
            export_params=True,
            opset_version=14,
            input_names=["input_ids", "pixel_values"],
            output_names=["logits"],
            
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "pixel_values": {0: "batch_size"},
                "decoder_input_ids": {0: "batch_size", 1: "decoder_sequence_length"},
                "logits": {0: "batch_size", 1: "decoder_sequence_length"}
            }
        )
        
        print(f"Модель сохранена в формате ONNX по пути: {onnx_save_path}")

    

###################################################################################

        # Загрузка ONNX модели
        onnx_model_path = "florence2_1.onnx"
        ort_session = ort.InferenceSession(onnx_model_path)

        # Загрузка процессора (для обработки входных данных)
        # processor = AutoProcessor.from_pretrained('path_to_model_checkpoint')

        # Пример подготовки данных
        # image = Image.open("path_to_image.jpg")
        # text_input = "Your task-specific text"


        # Подготовка входных данных
        # inputs = processor(text=text_input, images=image, return_tensors="pt")
        input_ids = inputs["input_ids"].numpy()  # Преобразуем в numpy для ONNX
        pixel_values = inputs["pixel_values"].numpy()
        dec = torch.tensor([[0]]).numpy()
        
        print(f"input_ids shape: {input_ids.shape}")
        print(f"pixel_values shape: {pixel_values.shape}")
        
        # Выполнение инференса с ONNX Runtime
        outputs = ort_session.run(
            None,
            {   "input.363": dec,
                "input_ids": input_ids,
                "pixel_values": pixel_values
            }
        )

        # Выходные данные
        logits = outputs[0]
        print("outputs[0]", outputs[0])
        print("len outputs", len(outputs))
        print("type outputs", type(outputs))
        
        
        generated_tokens_from_onnx = np.argmax(logits, axis=-1)
        print("Generated tokens from ONNX model:", generated_tokens_from_onnx)

        # Сохранение модели в формате TorchScript
        # torchscript_save_path = "florence2_scripted.pt"
        # scripted_model = torch.jit.trace(model, torch_example_inputs)
        # scripted_model.save(torchscript_save_path)
        # print(f"Модель сохранена в формате TorchScript по пути: {torchscript_save_path}")

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
        
        
        # Пример использования
        image_tensor = preprocess_image(image_path)
        # print(image_tensor.shape)  # torch.Size([1, 3, 768, 768])
        # print("image_tensor", image_tensor)
        # image_tensor для проверки моей предобработки

        # Запускаем модель
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        results, modified_image = my_run_example(task_prompt, image, text_input=text_input, image_tensor=image_tensor)

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
