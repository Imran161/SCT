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


# сохраняю дикомы в png

########## dcm_png_Sanya.py
def save_dicom_image_as_jpg(path2dicom_file, name, save_path):
    try:
        print("читаю:", path2dicom_file)
        ds = read_file(path2dicom_file)
        
        parts = path2dicom_file.split('/')
        # slice_name = parts[-1] # номер слайса добавил
        PatientName = str(ds.PatientName) # чтобы добавить в название имя пациента
        StudyInstanceUID = ds.StudyInstanceUID # UID добавлю еще
        # ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        img = apply_modality_lut(apply_voi_lut(ds.pixel_array, ds), ds)

        arr = 255.0*(img - np.min(img))/(np.max(img) - np.min(img))
        arr = cv2.merge([arr,arr,arr])
        if name==None:
            name=str(ds.StudyInstanceUID)
            if str(name)=="0":
                name = str(random.randint(0, 1000000000))
        asd = save_path + r"/" + name + "_" + PatientName  + "_" + StudyInstanceUID + '.png'

        cv2.imwrite(asd, arr)
    except Exception:
        traceback.print_exc()
        try:
            ds = read_file(path2dicom_file)
            ds.PlanarConfiguration = 0
            pixel_array = ds.pixel_array
            pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_BGR2RGB)
            if name == None:
                name = str(ds.StudyInstanceUID)
            asd = save_path + r"/" + name + PatientName + "_" + StudyInstanceUID + '.png'
            print(asd)
            cv2.imwrite(asd, pixel_array)
        except Exception:
            traceback.print_exc()
            print("ошибка на файле:", path2dicom_file, "переданное имя:", name, "save_path", save_path)

# рабочие строчки
# dcm_path = '100604476_data/1.2.392.200036.9116.2.6.1.48.1214245753.1506921033.705311/1.2.392.200036.9116.2.6.1.48.1214245753.1506921568.925536/001.dcm'
# save_dicom_image_as_jpg(dcm_path, "sanya_png1", "/home/imran/Документы/Innopolis/First_data_test/test_dicom_data")   

################ до сюда 



# хочу получить правильное название для файла чтобы потом save_dicom_image_as_jpg вызывать
dcm_path = "sct_project/sct_data/data/DICOM"

def dcm_search(dcm_path):
    for dirspaths,dirs,files in os.walk(dcm_path):
        for file in files:
            # img  = dcmread(dcm_path+"/"+file)
            print(f"dirspaths:{dirspaths}, dirs:{dirs}, files:{files}")
            img  = dcmread(dcm_path + "/" + file, force=False)
            print(img[0x10, 0x10]) # имя пациента надо в название записать
            # print("len:", len(img.PixelData))
            #print(img)
            # imid, img = prepare_image(img)
            print(type(img))
            print("shape:", img.shape)
            # save_img(img, '/home/imran/Документы/Innopolis/First_data_test/', 'first_test_from_notebook1')
            
            # plt.imshow(img, cmap = cm.Greys_r)
            break
    
    
# dcm_search(dcm_path)


def get_all_files(directory):
    all_files = []

    for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)

    return all_files

# Пример использования функции
# эти были
# directory = 'путь_к_директории'  # Укажите путь к вашей директории
# files_list = get_all_files(directory)

# for file_path in files_list:
#     print(file_path)




# эти три строки были
# all_files = get_all_files(dcm_path)

# print(len(all_files))
# print(all_files[0]) 

# вот такие имена файлов 
# 100604476_data/1.2.392.200036.9116.2.6.1.48.1214245753.1506921033.705311/1.2.392.200036.9116.2.6.1.48.1214245753.1506921568.925536/031.dcm

# теперь я могу передавать их первым аргументом в save_dicom_image_as_jpg

# надо теперь для второго аргумента имя правильное сделать
def get_slice_number(file):
    parts = file.split('/')
    slice = parts[-1].split('.')
    slice_number = slice[0]
    first = parts[1]
    first_parts = first.split('_')
    # first_number = first_parts[0]
    first_number = parts[4]
    return first_number, slice_number
    
# sct_project/sct_data/data/DICOM/191122139/1.2.392.200036.9116.2.6.1.48.1214245753.1574435249.81825/1.2.392.200036.9116.2.6.1.48.1214245753.1574452768.404242/025.dcm
# first_number sct

# print(get_slice_number(all_files[0])[1]) # 031.dcm 

# print('hello')


# вот имена директорий то есть это разные исследования
def get_direct_subdirectories(directory):
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return [os.path.join(directory, subdir) for subdir in subdirectories]

# Пример использования функции
directory = dcm_path  # Укажите путь к вашей директории
subdirectories_list = get_direct_subdirectories(directory)

for subdir_path in subdirectories_list:
    print("subdir_path", subdir_path)
# /home/imran-nasyrov/sct_project/sct_data/data/DICOM/191122139
    

for i in subdirectories_list:
    all_files = get_all_files(i)
    # print("i", i)
    for j in all_files:
        print("j", j)
        first_number, slice_number = get_slice_number(j)
        
        # print("first_number", first_number)
        # print("slice_number", slice_number)
        
        parts = j.split("/")
        last_number = parts[-2]
        # print("last_number", last_number)
        
        folder_path = f"/home/imran-nasyrov/sct_project/sct_data/FINAL_CONVERT/{first_number}/{last_number}/images"

        print("folder_path", folder_path)
        
        # Проверяем существование папки
        if not os.path.exists(folder_path):
            # Создаем папку и всех промежуточных родительских папок, если их нет
            os.makedirs(folder_path)
            print(f"Папка {folder_path} успешно создана")
        else:
            print(f"Папка {folder_path} уже существует")
            
        save_dicom_image_as_jpg( j, slice_number, f"/home/imran-nasyrov/sct_project/sct_data/FINAL_CONVERT/{first_number}/{last_number}/images") # перед images надо имя добавить в конце файла о нем написал

# все короче это четко работает


print("ok")