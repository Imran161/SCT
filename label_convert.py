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

from to_coco import numpy_mask_to_coco_polygon, create_coco_annotation_from_mask, create_empty_coco_annotation_from_mask



# сохраняю mha в json

# mha_path = "/home/imran/Документы/Innopolis/First_data_test/100604476_labels/1.2.392.200036.9116.2.6.1.48.1214245753.1506921033.705311/1.2.392.200036.9116.2.6.1.48.1214245753.1506921568.925536.mha"

mha_path = "sct_project/sct_data/labels/Labels/"

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
    # print("file", file) # sct_project/sct_data/labels/Labels/141015432/1.2.392.200036.9116.2.6.1.48.1214245753.1553254982.559318/1.2.392.200036.9116.2.6.1.48.1214245753.1553259815.239504.mha
    # first_number 141015432
    parts = file.split('/')
    first_number = parts[4]
    second_chislo = parts[6]
    return first_number, second_chislo

# subdirectories_list ['label_vse_papki/100604476_labels', 'label_vse_papki/120421792_labels']
# hello
# all_files ['label_vse_papki/100604476_labels/1.2.392.200036.9116.2.6.1.48.1214245753.1506921033.705311/1.2.392.200036.9116.2.6.1.48.1214245753.1506921568.925536.mha']
# all_files ['label_vse_papki/120421792_labels/1.2.392.200036.9116.2.6.1.48.1214245753.1579349065.852154/1.2.392.200036.9116.2.6.1.48.1214245753.1579349737.561083.mha']
# ok

def get_number_for_empty(file):
    # print("file", file) # sct_project/sct_data/FINAL_CONVERT_OLD/191122139/1.2.392.200036.9116.2.6.1.48.1214245753.1574452768.404242
    parts = file.split('/')
    first_number = parts[3]
    second_chislo = parts[4]
    return first_number, second_chislo

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
# print('subdirectories_list mha', subdirectories_list.index("sct_project/sct_data/labels/Labels/100604476"))

# # просто создал пустые папки которых не хватало
# numbers_mha = [int(item.split('/')[-1]) for item in subdirectories_list]

# subdirectories_list_dicom = get_direct_subdirectories("/home/imran-nasyrov/sct_project/sct_data/data/DICOM")
# # print("subdirectories_list dicom", subdirectories_list_mha)

# numbers_dicom = [int(item.split('/')[-1]) for item in subdirectories_list_dicom]
# print("numbers_dicom", len(numbers_dicom))
# print("numbers_mha", len(numbers_mha))
# result = [element for element in numbers_dicom if element not in numbers_mha]
# print("result", result)
# print("len result", len(result))
    
    
# result = [191122139, 190706973, 180919909, 170128014, 170100500, 190816693, 170220502, 200827252, 160828898, 200723385, 
#         170123329, 200114348, 190522739, 190106786, 170221794, 170207614, 200722868, 170120467, 190207693, 200404658, 
#         190608590, 201126613, 200318267, 121214647, 170207113, 161208699, 190807909] 
    
# base_path = "/home/imran-nasyrov/sct_project/sct_data/labels/Labels/"

# # Генерируем список путей к папкам
# folder_paths = [f"{base_path}{item}" for item in result]
    
    
# for folder_path in folder_paths:
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#         print(f"Создана папка: {folder_path}")
#     else:
#         print(f"Папка уже существует: {folder_path}")
    
# print("numbers_mha", len(numbers_mha))   
    
    
    
    
    
    
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
    
    # print(counts_non_zero)
    # print("values_list", values_list)
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




# сделаю чтобы сохранялись картинки с красными масками и классами

with tqdm(total=len(subdirectories_list), desc="Making files") as pbar_dirs: 
    for s in subdirectories_list:
        sub_subdirectories_list = get_direct_subdirectories(s)

        for i in sub_subdirectories_list: 
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
            
            # print("i", i) # sct_project/sct_data/labels/Labels/100604476
            # print("type", type(i)) # str
            ###############
            all_files = get_all_files(i)
            # print("all_files", all_files) ########## тут из-за того что я добавил пустые папки all_files пусто 
         
 
            # with tqdm(total=len(all_files), desc="Making files") as pbar:
            for j in all_files:
                first_number, second_chislo = get_slice_number(j)
                # print("first_number", first_number)
                # print("second_chislo", second_chislo)
                img = mha_read(j)
                # if i == "sct_project/sct_data/labels/Labels/100604476":
                #     print("j", j)
                #     print("img.shape[-1]", img.shape[-1]) # 601
                #     sys.exit()
                
                for index in range(img.shape[-1]):
                    slice_number = img[:,:,index]
                    # print("index", index)
                    # print("img.shape[-1]", img.shape[-1])
                    
                    # эти две строки чтобы потом в них 1сохранять дикомы с разметкой в виде картинок, но мне для сервера не надо это
                    # if not os.path.exists("/home/imran/Документы/Innopolis/First_data_test/test_mha_label/" + "jpg_" + first_number):
                    #     os.mkdir("/home/imran/Документы/Innopolis/First_data_test/test_mha_label/" + "jpg_" + first_number)
                    
                    # parts = first_number.split('/')
                    # print("parts", parts)
                    # first_number = parts[4]
                    
                    parts_second_chislo = second_chislo.split(".")
                    second_number = parts_second_chislo[:-1]
                    
                    result = '.'.join(second_number)
                    # print("result", result)
                        
                    # было так но мне надо сделать папку annotations и все такое поэтому написал код ниже
                    # dcm_pic_path = f"/home/imran/Документы/Innopolis/First_data_test/test_dicom_data/{first_number}_data"   ##### надо когда с тачки запускаюсь убрать слово _data
                    dcm_pic_path = f"/home/imran-nasyrov/sct_project/sct_data/FINAL_CONVERT/{first_number}/{result}/images"
                    # print("dcm_pic_path_pervyi", dcm_pic_path)
                    
                    
                    # Получаем список файлов в папке
                    file_list = os.listdir(dcm_pic_path)

                    # Фильтруем список файлов по индексу
                    # filtered_files = [file for file in file_list if file.startswith(str(index))] # надо поправить
                    
                    # if index < 10:
                    #     filtered_files = [file for file in file_list if file.startswith("00" + str(index+1))] 
                    #     # print("filtered_files",filtered_files)

                    # if index >= 10 and index < 100:
                    #     filtered_files = [file for file in file_list if file.startswith("0" + str(index+1))] 
                        
                    # if index >= 100:
                    #     filtered_files = [file for file in file_list if file.startswith(str(index+1))] 
                       
                    if index < 10:
                        filtered_files = [file for file in file_list if file.startswith("00" + str(index+1))] 
          
                    if index >= 10 and index < 100:
                        filtered_files = [file for file in file_list if file.startswith("0" + str(index+1))] 
                        
                    if index >= 100:
                        filtered_files = [file for file in file_list if file.startswith(str(index+1))] 
                            
                

                    # Проверяем, что найден хотя бы один файл
                    if filtered_files:
                        # Выбираем первый найденный файл
                        file_name = filtered_files[0]
                        full_file_path = os.path.join(dcm_pic_path, file_name)
                        
                        # Теперь у вас есть полный путь к выбранному файлу
                        # if i == "sct_project/sct_data/labels/Labels/100604476":
                        # print("Путь к выбранному файлу:", full_file_path)
                        # print("file_name", file_name)
                        
                        # Далее вы можете прочитать содержимое файла или выполнить другие операции с ним
                    else:
                        print("Файл с индексом", index, "не найден.")
                        # pass
                        
                # if i == "sct_project/sct_data/labels/Labels/100604476":
                #     print("file_list", file_list)
                #     print("len file_list", len(file_list))
                #     print("filtered_files", filtered_files)
                #     print("len filtered_files", len(filtered_files))
                #     sys.exit()  
                        
                        
                    image_id = f"{file_name}" # image_id = f"mha_image_{index}"
                    # print("image_id", image_id)
                    values = find_non_zero_label(slice_number) # было img
                    # print("slice_number.shape", slice_number.shape) # (512, 512)
                    # print("slice_number.shape[0]", slice_number.shape[0])
                    # print("slice_number.shape[1]", slice_number.shape[1])
                    # print("values", values)
                
                    # if values != 0:s
                    # заполняем annotations
                    # print("index", index)
                    # print("type(slice_number)", type(slice_number))
                    
                    #############
                    slice_number = cv2.rotate(slice_number, cv2.ROTATE_90_CLOCKWISE)
                    slice_number = cv2.flip(slice_number, 1)
                    ############
                    
                    image, annotation = create_coco_annotation_from_mask(slice_number, values, image_id) # index
                    # print("annotation here", annotation)
                    # print("annotation here", annotation["segmentation"])
                    if annotation["segmentation"] != [[0, 0, 0, 0, 0, 0]]:
                        # if len(coco_dataset["annotations"])
                        coco_dataset["annotations"].append(annotation) 
                    coco_dataset["images"].append(image) 
                    # coco_dataset["categories"].append(category) 
                    
                    
                    # image_from_polygon = draw_image_from_polygon(coco_dataset["annotations"][index]["segmentation"][0], slice_number.shape)
                    
                    
                    
                    
                    
                    # print("dcm_pic_path", dcm_pic_path)
                    dcm_img = cv2.imread(full_file_path)  
                    # image_with_contours = draw_contours_on_image(dcm_img, image_from_polygon, values) # тут надо указывать именно диком файл ##########

                    # cv2.imwrite(f"/home/imran/Документы/Innopolis/First_data_test/test_mha_label/jpg_{first_number}/mha_image_{index+1}.jpg", image_with_contours) #############
                    
                    
                    # cv2.imwrite(f"/home/imran/Документы/Innopolis/First_data_test/test_mha_label/jpg_{first_number}/{file_name}", image_with_contours)
                # до сюда if был
                
                # pbar.update(1)
                    
            # было так но мне надо сделать папку annotations и все такое поэтому написал код ниже
            # output_file = f"/home/imran/Документы/Innopolis/First_data_test/test_mha_label/json_{first_number}/mha_image_{first_number}.json"
            
            # вот тут first_number не существует когда get_all_files пусто, сделаю такой if может получится

 
            # try:
            folder_path = f"/home/imran-nasyrov/sct_project/sct_data/FINAL_CONVERT/{first_number}/{result}/annotations"
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
                        
            # output_file = f"/home/imran-nasyrov/sct_project/sct_data/FINAL_CONVERT/{first_number}/annotations/instances_default.json"
            output_file = f"{folder_path}/instances_default.json"
            
            with open(output_file, "w") as file:
                json.dump(coco_dataset, file)  
            # except:
            #     print("No") 
    
        
        
        
        pbar_dirs.update(1)          
#######################            


# файлы без патологий 
# directory_empty = "sct_project/sct_data/FINAL_CONVERT"
# subdirectories_list_empty = get_direct_subdirectories(directory_empty)
# with tqdm(total=len(subdirectories_list_empty), desc="Making files") as pbar_dirs: 
#     for s in subdirectories_list_empty:
#         sub_subdirectories_list_empty = get_direct_subdirectories(s)

#         for i in sub_subdirectories_list_empty: 
#             if 'annotations' not in os.listdir(i):
#                 # print("i", i) # 170207113, 190706973
#                 coco_dataset = {
#                 "info": {        
#                     "version": "",
#                     "date_created": "",
#                     "contributor": "",
#                     "year": 2024, # было ""
#                     "description": "",
#                     "url": ""
#                 },
                
#                 "licenses": [    
#                     {
#                         "url": "",
#                         "id": 0,
#                         "name": ""
#                     }
#                 ],
                
#                 "images": [],
#                 "annotations": [],
#                 # "categories": []
#                 "categories": [
#                     {
#                         "id": 0,
#                         "name": "0", # сделал тут имена строками и тогда в классе датасета заработало все
#                         "supercategory": ""
#                     },
#                     {
#                         "id": 1,
#                         "name": "1",
#                         "supercategory": ""
#                     },
#                     {
#                         "id": 2,
#                         "name": "2",
#                         "supercategory": ""
#                     },
#                     {
#                         "id": 3,
#                         "name": "3",
#                         "supercategory": ""
#                     },
#                     {
#                         "id": 4,
#                         "name": "4",
#                         "supercategory": ""
#                     },
#                     {
#                         "id": 5,
#                         "name": "5",
#                         "supercategory": ""
#                     }
#                 ]
#                 }
                
#                 # print("i", i) # sct_project/sct_data/labels/Labels/100604476
#                 # print("type", type(i)) # str
#                 ###############
#                 all_files = get_all_files(i)
#                 # print("all_files", all_files) ########## тут из-за того что я добавил пустые папки all_files пусто 
                
        
#                 first_number, second_chislo = get_number_for_empty(i)
#                 dcm_path = f"/home/imran-nasyrov/sct_project/sct_data/FINAL_CONVERT/{first_number}/{second_chislo}/images"
#                 # print("dcm_path", dcm_path) # /home/imran-nasyrov/sct_project/sct_data/FINAL_CONVERT/180423247/1.2.392.200036.9116.2.6.1.48.1214245753.1524218689.69145/images

#                 for j in all_files:
#                     # print("j", j)
#     # /home/imran-nasyrov/sct_project/sct_data/FINAL_CONVERT_OLD/191122139/1.2.392.200036.9116.2.6.1.48.1214245753.1574452768.404242/images/001_d93e766fca301f5700abe68b96b2e39f315509f0cfa487a45ed55ff6776644a0_1.2.392.200036.9116.2.6.1.48.1214245753.1574435249.81825.png
#                     image_name = j.split("images/")[1]
#                     # print("image_name", image_name)
#                     image = create_empty_coco_annotation_from_mask(image_name) # index
#                     coco_dataset["images"].append(image) 
                
                
#                 empty_folder_path = f"/home/imran-nasyrov/sct_project/sct_data/FINAL_CONVERT/{first_number}/{second_chislo}/annotations"
#                 print("empty_folder_path", empty_folder_path)
#                 if not os.path.exists(empty_folder_path):
#                     os.makedirs(empty_folder_path)
                            
#                 # output_file = f"/home/imran-nasyrov/sct_project/sct_data/FINAL_CONVERT/{first_number}/annotations/instances_default.json"
#                 output_file = f"{empty_folder_path}/instances_default.json"
                
#                 with open(output_file, "w") as file:
#                     json.dump(coco_dataset, file)  


print("ok")
