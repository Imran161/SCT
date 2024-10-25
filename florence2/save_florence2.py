import torch
import onnx

import os
import json
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import matplotlib.pyplot as plt  
import matplotlib.patches as patches  

path = "/mnt/netstorage/Medicine/Medical/florence2_files/outputs"


# Попытка загрузить все файлы, которые могут быть тензорами PyTorch
torch_tensors = []
for f in os.listdir(path):
    # print("f", f)
    file_path = os.path.join(path, f)
    # print("file_path", file_path)
    if os.path.isfile(file_path) and f.endswith('.pth'):
        # try:
        tensor = torch.load(file_path)
        # if isinstance(tensor, torch.Tensor):
        torch_tensors.append(tensor)
        # except Exception as e:
        #     print(f"Не удалось загрузить файл {f}: {e}")

print(torch_tensors)

for t in torch_tensors:
    print("t shape", t.shape)

print("len torch_tensors", len(torch_tensors))





