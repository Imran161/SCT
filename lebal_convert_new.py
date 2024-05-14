# попробую заново все сделать 

from pydicom import dcmread
from pydicom import read_file
from pydicom.pixel_data_handlers.util import apply_voi_lut
from pydicom.pixel_data_handlers.util import apply_modality_lut
import numpy as np
import os
import cv2
import joblib
import PIL
from glob import glob
import pydicom
import numpy as np
import pandas as pd
import os
import cv2
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from PIL import Image
import math
import seaborn as sns
from collections import defaultdict
from pathlib import Path
import cv2
from tqdm import tqdm
import re
import logging as l
from glob import glob
import argparse
import traceback
import random 
from medpy.io import load
from matplotlib import cm
import sys

from to_coco import numpy_mask_to_coco_polygon, create_coco_annotation_from_mask, coco_dataset


# сохраняю mha в json

# mha_path = "/home/imran/Документы/Innopolis/First_data_test/100604476_labels/1.2.392.200036.9116.2.6.1.48.1214245753.1506921033.705311/1.2.392.200036.9116.2.6.1.48.1214245753.1506921568.925536.mha"

mha_path = "/home/imran-nasyrov/sct_project/sct_data/labels/Labels/"

def mha_read(mha_path):
    # # вот тут сохранение mha в png но мне это не надо
    img, h = load(mha_path)
    # print (img.shape, img.dtype)
    
    # image = i[:,:,255]

    # image = (image - np.min(image)) / (np.max(image) - np.min(image))*255
    # image = image.astype(np.uint8)
    # rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # cv2.imwrite(f"mha_image_255.jpg", rgb_image) #################################################### надо правильное имя сделать 



    # преобразования в coco
    
    # тут надо имя правильное дать, но я наверное просто оставлю его как есть только номер слайса добвалю еще
    # image_id = "33.705311/1.2.392.200036.9116.2.6.1.48.1214245753.1506921568.925536/243.mha"
    # annotation = create_coco_annotation_from_mask(i[:,:,243:244], values[-1], image_id)
            
    return img
        
        
        
def get_all_files(directory):
    all_files = []

    for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)

    return all_files

def get_slice_number(file):
    print("file", file)
    parts = file.split('/')
    first_number = parts[7]
    second_chislo = parts[6]
    return first_number, second_chislo

# subdirectories_list ['label_vse_papki/100604476_labels', 'label_vse_papki/120421792_labels']
# hello
# all_files ['label_vse_papki/100604476_labels/1.2.392.200036.9116.2.6.1.48.1214245753.1506921033.705311/1.2.392.200036.9116.2.6.1.48.1214245753.1506921568.925536.mha']
# all_files ['label_vse_papki/120421792_labels/1.2.392.200036.9116.2.6.1.48.1214245753.1579349065.852154/1.2.392.200036.9116.2.6.1.48.1214245753.1579349737.561083.mha']
# ok



all_files = get_all_files(mha_path)
# print("all_files", all_files)
# print(get_slice_number(all_files)) 


# вот имена директорий то есть это разные исследования
def get_direct_subdirectories(directory):
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return [os.path.join(directory, subdir) for subdir in subdirectories]

# Пример использования функции
directory = mha_path  # Укажите путь к вашей директории
subdirectories_list = get_direct_subdirectories(directory)
# print('subdirectories_list', subdirectories_list)

# for subdir_path in subdirectories_list:
#     print("subdir_path", subdir_path)
    
    
# print("hello")
# фигачу для всех папок
# вот это все надо но я щас уберу тк мне надо цветные маски сохранить
# for i in subdirectories_list:
#     all_files = get_all_files(i)
#     # print("all_files", all_files)
#     for j in all_files:
#         first_number = get_slice_number(j)
#         img = mha_read(j)
        
#         # print("j", j)
#         # print("img.shape[-1]", img.shape[-1])
        
#         for index in range(img.shape[-1]): 
            
#             if not os.path.exists("/home/imran/Документы/Innopolis/First_data_test/test_mha_label/" + first_number):
#                 os.mkdir("/home/imran/Документы/Innopolis/First_data_test/test_mha_label/" + first_number)
#             slice_number = img[:,:,index]
#             # print("index", index)
#             slice_number = (slice_number - np.min(slice_number)) / (np.max(slice_number) - np.min(slice_number))*255
#             slice_number = slice_number.astype(np.uint8)
#             rgb_image = cv2.cvtColor(slice_number, cv2.COLOR_GRAY2RGB)

#             # cv2.imwrite(f"/home/imran/Документы/Innopolis/First_data_test/test_mha_label/{first_number}/mha_image_{index}.jpg", rgb_image) 
#             # это сохранит все в png но нам это не надо надо в json зафигачить все 
# вот до сюда надо



# найдем индексы не нулевых масок это чтобы Сане показать потом
# было так
# def find_non_zero_label(img):
#     values_list = set()
#     counts_non_zero = 0

#     for index in range(img.shape[-1]): 
#         values, counts = np.unique(img[:,:,index], return_counts=True)
#         if values.shape[0] > 1:
#             values_list.update(values)
#             counts_non_zero += 1
#     values_list = list(values_list)
    
#     print(counts_non_zero)
#     print("values_list", values_list)
#     return values_list[-1]
#     # values_list[-1] это и есть класс патологии ################################################ надо дальше будет использовать
    
    
def find_non_zero_label(img):
    values_list = set()
    values_list.add(0)
    counts_non_zero = 0

    # for index in range(img.shape[-1]): 
    values, counts = np.unique(img, return_counts=True)
    if values.shape[0] > 1:
        values_list.update(values)
        counts_non_zero += 1
    values_list = list(values_list)
    
    print(counts_non_zero)
    print("values_list", values_list)
    if len(values_list) != 0:
        return values_list[-1]
    return 0
    # values_list[-1] это и есть к
    
    
def draw_image_from_polygon(polygon, image_shape):
    image = np.zeros(image_shape, dtype=np.uint8)
    
    # red_mask = np.zeros_like(image)
    # red_mask[:] = (0, 0, 255)  # Красный цвет (BGR форма  т)

    # # Наложение красной маски на изображение pic
    # result = cv2.addWeighted(image, 1, red_mask, 0.5, 0)
    
    polygon = np.array(polygon).reshape((-1, 2)).astype(np.int32)
    cv2.fillPoly(image, [polygon], (255, 255, 255))
    return image


def draw_contours_on_image(image, mask, label):
    
    # Поверну маску 
    # mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    # mask = cv2.flip(mask, 1)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Рисуем контуры на изображении
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 0, 255), 2)
    
    # Добавляем надпись "label class: значение" к изображению
    # plt.text(10, 20, f"label class: {label}", color='red', fontsize=12, fontweight='bold')
    cv2.putText(image_with_contours, f"label class: {label}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


    return image_with_contours







print("hello")
# сделаю чтобы сохранялись картинки с красными масками и классами
for i in subdirectories_list:
    coco_dataset = {
        "info": {        
            "version": "",
            "date_created": "",
            "contributor": "",
            "year": 2024, # было ""
            "description": "",
            "url": ""
        },
        
        "licenses": [    
            {
                "url": "",
                "id": 0,
                "name": ""
            }
        ],
        
        "images": [],
        "annotations": [],
        # "categories": []
        "categories": [
            {
                "id": 0,
                "name": "0", # сделал тут имена строками и тогда в классе датасета заработало все
                "supercategory": ""
            },
            {
                "id": 1,
                "name": "1",
                "supercategory": ""
            },
            {
                "id": 2,
                "name": "2",
                "supercategory": ""
            },
            {
                "id": 3,
                "name": "3",
                "supercategory": ""
            },
            {
                "id": 4,
                "name": "4",
                "supercategory": ""
            },
            {
                "id": 5,
                "name": "5",
                "supercategory": ""
            }
        ]
    }
    
    print("i", i)
    all_files, second_chislo = get_all_files(i)
    print("all_files", all_files)
    for j in all_files:
        first_number = get_slice_number(j)
        print("first number here", first_number)
        print("second_chislo", second_chislo)
    #     img = mha_read(j)
        
    #     print("j", j)
    #     print("img.shape[-1]", img.shape[-1]) # 601
        
    #     for index in range(img.shape[-1]):
    #         # было так но мне надо сделать папку annotations и все такое поэтому написал код ниже
    #         # if not os.path.exists("/home/imran/Документы/Innopolis/First_data_test/test_mha_label/" + "json_" + first_number):
    #         #     os.mkdir("/home/imran/Документы/Innopolis/First_data_test/test_mha_label/" + "json_" + first_number)
            
    #         # только эти две строки по ходу не нужны вообще
    #         # if not os.path.exists("/home/imran/Документы/Innopolis/First_data_test/FINAL_CONVERT/" + first_number):
    #         #     os.mkdir("/home/imran/Документы/Innopolis/First_data_test/FINAL_CONVERT/" + first_number)               
                
    #         slice_number = img[:,:,index]
    #         print("index", index)
            
    #         # эти две строки чтобы потом в них 1сохранять дикомы с разметкой в виде картинок, но мне для сервера не надо это
    #         # if not os.path.exists("/home/imran/Документы/Innopolis/First_data_test/test_mha_label/" + "jpg_" + first_number):
    #         #     os.mkdir("/home/imran/Документы/Innopolis/First_data_test/test_mha_label/" + "jpg_" + first_number)
            
    #         parts = first_number.split('_')
    #         first_number = parts[0]
    #         # было так но мне надо сделать папку annotations и все такое поэтому написал код ниже
    #         # dcm_pic_path = f"/home/imran/Документы/Innopolis/First_data_test/test_dicom_data/{first_number}_data"   ##### надо когда с тачки запускаюсь убрать слово _data
    #         dcm_pic_path = f"/home/imran-nasyrov/sct_project/sct_data/FINAL_CONVERT/{first_number}/images"
            
    #         print("dcm_pic_path_pervyi", dcm_pic_path)
            
    #                         # Получаем список файлов в папке
    #         file_list = os.listdir(dcm_pic_path)

    #         # Фильтруем список файлов по индексу
    #         # filtered_files = [file for file in file_list if file.startswith(str(index))] # надо поправить
            
    #         if index < 10:
    #             filtered_files = [file for file in file_list if file.startswith("00" + str(index+1))] 
    #             print("filtered_files",filtered_files)

    #         if index >= 10 and index < 100:
    #             filtered_files = [file for file in file_list if file.startswith("0" + str(index+1))] 
                
    #         if index >= 100:
    #             filtered_files = [file for file in file_list if file.startswith(str(index+1))] 
                

    #         # Проверяем, что найден хотя бы один файл
    #         if filtered_files:
    #             # Выбираем первый найденный файл
    #             file_name = filtered_files[0]
    #             full_file_path = os.path.join(dcm_pic_path, file_name)
                
    #             # Теперь у вас есть полный путь к выбранному файлу
    #             print("Путь к выбранному файлу:", full_file_path)
    #             print("file_name", file_name)
                
    #             # Далее вы можете прочитать содержимое файла или выполнить другие операции с ним
    #         else:
    #             print("Файл с индексом", index, "не найден.")
                
                
                
                
    #         image_id = f"{file_name}" # image_id = f"mha_image_{index}"
    #         print("image_id", image_id)
    #         values = find_non_zero_label(slice_number) # было img
    #         print("slice_number.shape", slice_number.shape) # (512, 512)
    #         print("slice_number.shape[0]", slice_number.shape[0])
    #         print("slice_number.shape[1]", slice_number.shape[1])
    #         print("values", values)
         
    #         # if values != 0:s
    #         # заполняем annotations
    #         print("index", index)
    #         print("type(slice_number)", type(slice_number))
            
    #         #############
    #         slice_number = cv2.rotate(slice_number, cv2.ROTATE_90_CLOCKWISE)
    #         slice_number = cv2.flip(slice_number, 1)
    #         ############
            
    #         image, annotation = create_coco_annotation_from_mask(slice_number, values, image_id) # index
    #         print("annotation here", annotation)
    #         print("annotation here", annotation["segmentation"])
    #         if annotation["segmentation"] != [[0, 0, 0, 0, 0, 0]]:
    #             coco_dataset["annotations"].append(annotation) 
    #         coco_dataset["images"].append(image) 
            
    #         # coco_dataset["categories"].append(category) 
            
            
    #         # image_from_polygon = draw_image_from_polygon(coco_dataset["annotations"][index]["segmentation"][0], slice_number.shape)
            
            
            
            
            
    #         # print("dcm_pic_path", dcm_pic_path)
    #         dcm_img = cv2.imread(full_file_path)  
    #         # image_with_contours = draw_contours_on_image(dcm_img, image_from_polygon, values) # тут надо указывать именно диком файл ##########

    #         # cv2.imwrite(f"/home/imran/Документы/Innopolis/First_data_test/test_mha_label/jpg_{first_number}/mha_image_{index+1}.jpg", image_with_contours) #############
            
            
    #         # cv2.imwrite(f"/home/imran/Документы/Innopolis/First_data_test/test_mha_label/jpg_{first_number}/{file_name}", image_with_contours)
    #     # до сюда if был
        
    # # было так но мне надо сделать папку annotations и все такое поэтому написал код ниже
    # # output_file = f"/home/imran/Документы/Innopolis/First_data_test/test_mha_label/json_{first_number}/mha_image_{first_number}.json"
    
    # # if i == "label_vse_papki/100604476_labels":
    # #     print("len coco_dataset[images]", len(coco_dataset["images"]))
    # #     sys.exit()
    # try:
    #         if not os.path.exists("/home/imran-nasyrov/sct_project/sct_data/FINAL_CONVERT/" + first_number + "/annotations"):
    #             os.mkdir("/home/imran-nasyrov/sct_project/sct_data/FINAL_CONVERT/" + first_number + "/annotations")      
                    
    #         output_file = f"/home/imran-nasyrov/sct_project/sct_data/FINAL_CONVERT/{first_number}/annotations/instances_default.json"
            
    #         with open(output_file, "w") as file:
    #             json.dump(coco_dataset, file)  
    # except:
    #     print("No")        

######################################## короооооче #########  у меня несколько исследований для одной папки есть
# а я это не учел нафиг и они все перемешались е мое надо это исправлять 


print("ok")
