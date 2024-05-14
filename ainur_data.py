import h5py
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from tqdm import tqdm

from to_coco import numpy_mask_to_coco_polygon, create_coco_annotation_from_mask
import json


def h5py_read(path, in_channels):
    with h5py.File(path, "r") as f:
        key_string = list(f.keys())[0]
        assert str(int(key_string)) == key_string

        class_ids = f["class_ids"][:]
        data = f[key_string]

        slide_name = f[key_string + "_slide_name"][()].decode("utf-8")
        roi_xywh = f[key_string + "_roi_xywh"][:]

        image = data[:, :, :in_channels]
        if image.dtype == "uint8":
            image = (image / 255).astype("float32")

        mask = data[:, :, in_channels:]
        # if (s := key_string + "_contour_mask") in f:
        #     contour_mask = f[s][:]
        assert len(roi_xywh) == 4

    assert len(image.shape) == len(mask.shape)
    assert image.shape[:2] == mask.shape[:2]

    target = class_ids[0]

    return {
        "image": image,
        "target": int(target),
        "slide_name": slide_name,
        "roi_xywh": roi_xywh,
        "mask": mask,
    }


def draw_contours_on_image(image, mask, label):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 0, 255), 2)

    cv2.putText(
        image_with_contours,
        f"label class: {label}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    return image_with_contours


def draw_image_from_polygon(polygon, image_shape):
    image = np.zeros(image_shape, dtype=np.uint8)

    # red_mask = np.zeros_like(image)
    # red_mask[:] = (0, 0, 255)  # Красный цвет (BGR форма  т)

    # # Наложение красной маски на изображение pic
    # result = cv2.addWeighted(image, 1, red_mask, 0.5, 0)

    polygon = np.array(polygon).reshape((-1, 2)).astype(np.int32)
    cv2.fillPoly(image, [polygon], (255, 255, 255))
    return image


def get_direct_subdirectories(directory):
    subdirectories = [
        d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))
    ]
    return [os.path.join(directory, subdir) for subdir in subdirectories]


def get_all_files(directory):
    all_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)

    return all_files


def get_slice_number(file):
    # j small_ainur/validate/home_ainur-karimov_data_raw_04.07.22_Размеченные_22.2.4739_level_0_78.hdf5
    # j small_ainur/validate/home_ainur-karimov_data_raw_29.01.23_Датасеты биопсии C16_22.1.2205-1_22.1.2205-1_level_0_66.hdf5#

    # вот так попробую чтобы было 04.07.22_Размеченные_22.2.4739
    parts = file.split("/")
    slice = parts[-1].split("_")
    first_number = "_".join(slice[-3:])  # было так для желудка 
    # first_number = slice[4] # щас так 
    slice_number = first_number.split(".")[0]

    return first_number, slice_number
  
  
if __name__ == "__main__":
    in_channels = 3
    
    # это для одного файлв
    # path_to_hdf5 = "/home/imran/Документы/Innopolis/First_data_test/small_ainur/validate/home_ainur-karimov_data_raw_29.01.23_Датасеты биопсии C16_22.1.2205-1_22.1.2205-1_level_0_108.hdf5"
    # h5_dict = h5py_read(path_to_hdf5, in_channels)

    # image = h5_dict["image"]
    # target = h5_dict["target"]
    # mask = h5_dict["mask"][:, :, target]
    # image_with_contours = draw_contours_on_image(image, mask, target)
    # cv2.imshow("Contours", image_with_contours)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # вот тут для всех сделаю
    
    # тут убрал полный путь чтобы пути к файлам были короче
    # directory_path = "sct_project/sct_data/ainur_paths"
    directory_path = "/mnt/datastore/Medical/stomach_paths"
    subdirectories_list = get_direct_subdirectories(directory_path)

    
    with tqdm(total=len(subdirectories_list), desc="Making files") as pbar_dirs:      
        # print("subdirectories_list", subdirectories_list) # тут test, train, validate
        for i in subdirectories_list: # tqdm, desc="Processing subdirectories"):
            # print("i", i) # sct_project/sct_data/ainur_paths/29.01.23_Датасеты биопсии C16_22.2.15881_22.2.15881
            
            # try: # добавил потому что файл какой то не читался, возможно скачивание не удачно оборвал
                
            coco_dataset = {
                        "info": {
                            "version": "",
                            "date_created": "",
                            "contributor": "",
                            "year": 2024,  # было ""
                            "description": "",
                            "url": "",
                        },
                        "licenses": [{"url": "", "id": 0, "name": ""}],
                        "images": [],
                        "annotations": [],
                        # "categories": []
                        "categories": [
                            
                            # так что вроде теперь так для желудка
                            { "id": 24, "name": "0", "supercategory": "" },
                            { "id": 0, "name": "SE", "supercategory": "" },
                            { "id": 1, "name": "GT", "supercategory": "" },
                            { "id": 2, "name": "NG", "supercategory": "" },
                            { "id": 3, "name": "F", "supercategory": "" },
                            { "id": 4, "name": "IM", "supercategory": "" },
                            { "id": 5, "name": "LT", "supercategory": "" },
                            { "id": 6, "name": "GINL", "supercategory": "" },
                            { "id": 7, "name": "GINH", "supercategory": "" },
                            { "id": 8, "name": "TACG1", "supercategory": "" },
                            { "id": 9, "name": "TACG2", "supercategory": "" },
                            { "id": 10, "name": "TACG3", "supercategory": "" },
                            { "id": 11, "name": "PACG1", "supercategory": "" },
                            { "id": 12, "name": "PACG2", "supercategory": "" },
                            { "id": 13, "name": "MPAC", "supercategory": "" },
                            { "id": 14, "name": "PCC", "supercategory": "" },
                            { "id": 15, "name": "PCC-NOS", "supercategory": "" },
                            { "id": 16, "name": "MAC1", "supercategory": "" },
                            { "id": 17, "name": "MAC2", "supercategory": "" },
                            { "id": 18, "name": "ACLS", "supercategory": "" },
                            { "id": 19, "name": "HAC", "supercategory": "" },
                            { "id": 20, "name": "ACFG", "supercategory": "" },
                            { "id": 21, "name": "SCC", "supercategory": "" },
                            { "id": 22, "name": "NDC", "supercategory": "" },
                            { "id": 23, "name": "NED", "supercategory": "" }

                            # вот так было для желудка но потом выяснилось что надо сделать перестановку вниз на один
                            # # это старые для желудка
                            # {
                            #     "id": 0,
                            #     "name": "0",  
                            #     "supercategory": "",
                            # }
                            # {"id": 1, "name": "SE", "supercategory": ""},
                            # {"id": 2, "name": "GT", "supercategory": ""},
                            # {"id": 3, "name": "NG", "supercategory": ""},
                            # {"id": 4, "name": "F", "supercategory": ""},
                            # {"id": 5, "name": "IM", "supercategory": ""},
                            
                            # {"id": 6, "name": "LT", "supercategory": ""},
                            # {"id": 7, "name": "GINL", "supercategory": ""},
                            # {"id": 8, "name": "GINH", "supercategory": ""},
                            # {"id": 9, "name": "TACG1", "supercategory": ""},
                            # {"id": 10, "name": "TACG2", "supercategory": ""},
                            
                            # {"id": 11, "name": "TACG3", "supercategory": ""},
                            # {"id": 12, "name": "PACG1", "supercategory": ""},
                            # {"id": 13, "name": "PACG2", "supercategory": ""},
                            # {"id": 14, "name": "MPAC", "supercategory": ""},
                            # {"id": 15, "name": "PCC", "supercategory": ""},
                            
                            # {"id": 16, "name": "PCC-NOS", "supercategory": ""},
                            # {"id": 17, "name": "MAC1", "supercategory": ""},
                            # {"id": 18, "name": "MAC2", "supercategory": ""},
                            # {"id": 19, "name": "ACLS", "supercategory": ""},
                            # {"id": 20, "name": "HAC", "supercategory": ""},
                            
                            # {"id": 21, "name": "ACFG", "supercategory": ""},
                            # {"id": 22, "name": "SCC", "supercategory": ""},
                            # {"id": 23, "name": "NDC", "supercategory": ""},
                            # {"id": 24, "name": "NED", "supercategory": ""}
                            
                            
                            # это новые для кишки
                            
                            # возможно надо нулевой добавить для пустышек
                            # {"id": 1, "name": "GT", "supercategory": ""},
                            # {"id": 2, "name": "NGC", "supercategory": ""},
                            # {"id": 3, "name": "F", "supercategory": ""},
                            # {"id": 4, "name": "LT", "supercategory": ""},
                            # {"id": 5, "name": "SDL", "supercategory": ""},
                            
                            # {"id": 6, "name": "SDH", "supercategory": ""},
                            # {"id": 7, "name": "HPM", "supercategory": ""},
                            # {"id": 8, "name": "HPG", "supercategory": ""},
                            # {"id": 9, "name": "APL", "supercategory": ""},
                            # {"id": 10, "name": "APH", "supercategory": ""},
                            
                            # {"id": 11, "name": "TA", "supercategory": ""},
                            # {"id": 12, "name": "VA", "supercategory": ""},
                            # {"id": 13, "name": "INL", "supercategory": ""},
                            # {"id": 14, "name": "INH", "supercategory": ""},
                            # {"id": 15, "name": "ADCG1", "supercategory": ""},
                            
                            # {"id": 16, "name": "ADCG2", "supercategory": ""},
                            # {"id": 17, "name": "ADCG3", "supercategory": ""},
                            # {"id": 18, "name": "MAC", "supercategory": ""},
                            # {"id": 19, "name": "SRC", "supercategory": ""},
                            # {"id": 20, "name": "MC", "supercategory": ""},
                            
                            # {"id": 21, "name": "NDC", "supercategory": ""},
                            # {"id": 22, "name": "NED", "supercategory": ""}
                            
                        ] # была запятая 
                    }
            
            all_files = get_all_files(i)
            with tqdm(total=len(all_files), desc="Making files") as pbar:
                for j in all_files: # tqdm, desc=f"Processing files in {i}"):
                    # print("j", j)
                    # пока сделаю чтобы у каждой картинки был свой json чтобы Сане отправить
                    


                    path_to_hdf5 = j
                    h5_dict = h5py_read(path_to_hdf5, in_channels)

                    h5_image = h5_dict["image"]
                    target = h5_dict["target"]
                    mask = h5_dict["mask"][:, :, target]
                    # image_with_contours = draw_contours_on_image(image, mask, target) вот по сути картинка готовая но нужно через coco сделать
                    # пока так сделаю для простоты
                    # print("image", image)
                    # print("h5_image shape", h5_image.shape)
                    # print("mask", mask)
                    # print("target", target) # 14
                    # print("image_id", image_id) # small_ainur/validate/home_ainur-karimov_data_raw_04.07.22_Размеченные_22.2.4739_level_0_67.hdf5
                    
                    first_number, slice_number = get_slice_number(j)
                    # print("first_number", first_number) # first_number level_0_180.hdf5
                    # print("slice_number", slice_number) # slice_number level_0_180
                    image_id = slice_number
                    
                    # parts = image_id.split('_')
                    # id_number = parts[-1]
                    # id_number = int(id_number)
                    # print("id_number", id_number) # 180
                    
                    # это фотки сохранит

                    papka = i.split("/")[-1]
                    # print("papka", papka) # 29.01.23_Датасеты биопсии C16_22.2.15881_22.2.15881 
    
                    # images_path = f"/home/imran-nasyrov/sct_project/sct_data/CONVERT_AINUR/{papka}/images"
                    images_path = f"/mnt/datastore/Medical/stomach_json/{papka}/images"
                    if not os.path.exists(images_path):
                        os.makedirs(images_path)
                    
                    result = cv2.normalize(h5_image, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    pixel_array = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                    out_convert_image = f"{images_path}/{image_id}.jpg"
                    # print("out_convert_image", out_convert_image) # /home/imran-nasyrov/sct_project/sct_data/CONVERT_AINUR/29.01.23_Датасеты биопсии C16_200170-1-1_200170-1-1/images/level_0_1356.jpg
                    
                    cv2.imwrite(out_convert_image, pixel_array)
                        
                    image_name = out_convert_image.split("/")[-1]
                    # print("image_name", image_name)
                    
                    image, annotation = create_coco_annotation_from_mask(mask, target, image_name) # вместо image_id мне надо передавать имя файла jpg поэтому создам его выше
                    # print("annotation here", annotation)
                    # print("annotation here", len(annotation["segmentation"])) # тут нормально все все области хранятся
                    total_length = sum(len(sublist) for sublist in annotation["segmentation"])
                    # print("total_length here", total_length)
                    if annotation["segmentation"] != [[0, 0, 0, 0, 0, 0]]:
                        # if len(coco_dataset["annotations"])
                        coco_dataset["annotations"].append(annotation) 
                    coco_dataset["images"].append(image) 
                    
                    # для неразмеченных тоже json создаются но с пустым annotations = []
                    
                    
                    # total_length = sum(len(sublist) for sublist in annotation["segmentation"])
                    # print("total_length here", total_length)
                    
                    # тут короче опять есть неразмеченные картинки
                    # Это сохранение картинок моим кодом
                    # try:
                    #     # print("coco_dataset[annotations]", coco_dataset["annotations"]) # 1
                    #     # print("coco_dataset[annotations][0]", coco_dataset["annotations"][0])
                    #     image_from_polygon = draw_image_from_polygon(coco_dataset["annotations"][0]["segmentation"][0], mask.shape)
                    # except:
                    #     print("no")
                    # my_result = cv2.normalize(h5_image, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    # my_pixel_array = cv2.cvtColor(my_result, cv2.COLOR_RGB2BGR)
                    
                    # image_with_contours = draw_contours_on_image(my_pixel_array, image_from_polygon, target) # тут надо указывать именно диком файл ##########
                    
            
                

                    # Пути к папке, которую нужно создать
                    
                    # folder_path = f"/home/imran-nasyrov/sct_project/sct_data/output_image_ainur/{target}"

                    # # Проверяем существование папки
                    # if not os.path.exists(folder_path):
                    #     # Создаем папку и всех промежуточных родительских папок, если их нет
                    #     os.makedirs(folder_path)
                    #     print(f"Папка {folder_path} успешно создана")
                    # else:
                    #     print(f"Папка {folder_path} уже существует")


                    
                    
                    # my_folder_path = f"/home/imran-nasyrov/sct_project/sct_data/my_output_image_ainur/{target}"
                
                    # if not os.path.exists(my_folder_path):
                    #     os.makedirs(my_folder_path)
                
                    # my_result = cv2.normalize(image_with_contours, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    # my_pixel_array = cv2.cvtColor(my_result, cv2.COLOR_RGB2BGR)
                    # out_convert_image = f"{images_path}/{id_number}.jpg"
                    
                    # cv2.imwrite(f"{my_folder_path}/{id_number}.jpg", my_pixel_array)
                    
                    # папки с json создал, осталось картинки добавить
                    # try:
                    # folder_path = f"/home/imran-nasyrov/sct_project/sct_data/CONVERT_AINUR/{papka}/annotations"
                    folder_path = f"/mnt/datastore/Medical/stomach_json/{papka}/annotations"
                    
                    
                    # images_path = f"/home/imran-nasyrov/sct_project/sct_data/CONVERT_AINUR/{id_number}/images"
                    
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    
                    # if not os.path.exists(images_path):
                    #     os.makedirs(images_path)
                        
                    # это надо будет когда класс вызывать буду
                    # all_out_files = get_all_files(f"/home/imran-nasyrov/sct_project/sct_data/CONVERT_AINUR/{id_number}")
                    # print("all_out_files", all_out_files)
                    # if len(all_out_files) < 20:
                    # plt.imshow(h5_image)
                    # plt.axis('off')  
                    # plt.savefig(f"{images_path}/{id_number}.jpg", bbox_inches='tight', pad_inches=0)
                    
                    # result = cv2.normalize(h5_image, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    # pixel_array = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

                    # cv2.imwrite(f"{images_path}/{id_number}.jpg", pixel_array)
#################################                       
                    output_file = f"{folder_path}/instances_default.json"
                    
                    with open(output_file, "w") as file:
                        json.dump(coco_dataset, file)  
                    # except:
                    #     print("No") 
                    
                    pbar.update(1)
                    
            pbar_dirs.update(1)  
        
        # except:
        #     pass
#########################################
# j small_ainur/validate/home_ainur-karimov_data_raw_04.07.22_Размеченные_22.2.4739_level_0_78.hdf5
# j small_ainur/validate/home_ainur-karimov_data_raw_29.01.23_Датасеты биопсии C16_22.1.2205-1_22.1.2205-1_level_0_66.hdf5#


    print("ok")
