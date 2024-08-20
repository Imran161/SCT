import os
from shutil import copy

import cv2
import numpy as np
from tqdm import tqdm


def get_all_files(directory):
    all_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)

    return all_files


def get_direct_subdirectories(directory):
    subdirectories = [
        d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))
    ]
    return [os.path.join(directory, subdir) for subdir in subdirectories]


def get_slice_number(file):
    # print("file", file)
    # parts = file.split("/")
    # slice = parts[-1].split("_")
    # # first_number = slice[6] # вот такие бывают [001] 22.2.5255, 22.2.13327, 2191-002, 76003-05 2021 получилось 319 штук
    # # slice[4] 04.07.22
    # first_number = slice[-4]
    # slice_number = slice[0]# slice[3]  # raw
    # # если по raw сплитить то одна папка получается
    # # по 04.07.22 всего две папки

    # старый желудок
    # вот так попробую чтобы было 04.07.22_Размеченные_22.2.4739

    # новый желужок
    # file /mnt/datastore/WRITE_ACCESS_DIR/patho/cropped/stomach/1/256/train/29.01.23_Датасеты биопсии C16_1002989_1002989_level_0_67.hdf5
    parts = file.split("/")
    slice = parts[-1].split("_")
    # first_number = slice[4]
    # было так для старых данных по желудку теперь сделаю строкой выше ######################################################################
    # first_number = "_".join(slice[4:-3])  # для старого желудка так было
    first_number = "_".join(
        slice[:-3]
    )  # для нового так 29.01.23_Датасеты биопсии C16_1002989_1002989
    slice_number = slice[-2]  # Берем предпоследний элемент

    return first_number, slice_number


def draw_image_from_polygon(polygon, image_shape):
    image = np.zeros(image_shape, dtype=np.uint8)

    # red_mask = np.zeros_like(image)
    # red_mask[:] = (0, 0, 255)  # Красный цвет (BGR форма  т)

    # # Наложение красной маски на изображение pic
    # result = cv2.addWeighted(image, 1, red_mask, 0.5, 0)

    polygon = np.array(polygon).reshape((-1, 2)).astype(np.int32)
    cv2.fillPoly(image, [polygon], (255, 255, 255))
    return image


# Ваши существующие функции идут здесь...


def process_files(all_files, new_dirs):
    with tqdm(total=len(all_files), desc="Copying files") as pbar:
        for j in all_files:
            first_number, slice_number = get_slice_number(j)
            folder_path = f"{new_dirs}/{first_number}"

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            copy(j, folder_path)
            pbar.update(1)


if __name__ == "__main__":
    # то же самое по времени получилось

    # new_dirs = "/home/imran-nasyrov/sct_project/sct_data/ainur_paths"
    # directory_path = "ainur_data"
    # subdirectories_list = get_direct_subdirectories(directory_path)

    # num_cores = os.cpu_count()
    # print("Количество доступных ядер процессора:", num_cores)

    # with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    #     futures = []
    #     for subdir in subdirectories_list:
    #         all_files = get_all_files(subdir)

    #         futures.append(executor.submit(process_files, all_files, new_dirs))

    #     for future in concurrent.futures.as_completed(futures):
    #         future.result()

    # было так но поопробую с тредами
    #########################
    # new_dirs = "/home/imran-nasyrov/sct_project/sct_data/ainur_paths"
    # new_dirs = "/mnt/datastore/Medical/guts_paths"
    new_dirs = "/mnt/datastore/Medical/stomach_paths"

    # тут убрал полный путь чтобы пути к файлам были короче
    # directory_path = "ainur_data"
    # directory_path = "/mnt/datastore/WRITE_ACCESS_DIR/patho/cropped/colon/0/256/" # кишки
    directory_path = (
        "/mnt/datastore/WRITE_ACCESS_DIR/patho/cropped/stomach/1/256/"  # желудок
    )
    subdirectories_list = get_direct_subdirectories(directory_path)

    # print("subdirectories_list", subdirectories_list) # тут test, train, validate
    for i in subdirectories_list:
        all_files = get_all_files(i)

        with tqdm(total=len(all_files), desc="Copying files") as pbar:
            # print("all_files", all_files)
            for j in all_files:
                # пока сделаю чтобы у каждой картинки был свой json чтобы Сане отправить

                first_number, slice_number = get_slice_number(j)
                # print("first_number", first_number)
                # print("slice_number", slice_number)

                folder_path = f"{new_dirs}/{first_number}"

                # Проверяем существование папки
                if not os.path.exists(folder_path):
                    # Создаем папку и всех промежуточных родительских папок, если их нет
                    os.makedirs(folder_path)
                    # print(f"Папка {folder_path} успешно создана")
                # else:
                # print(f"Папка {folder_path} уже существует")

                copy(j, folder_path)

                pbar.update(1)
################################
# а были вот такие в вале, трейн я тогда не видел
# j small_ainur/validate/home_ainur-karimov_data_raw_04.07.22_Размеченные_22.2.4739_level_0_78.hdf5
# j small_ainur/validate/home_ainur-karimov_data_raw_29.01.23_Датасеты биопсии C16_22.1.2205-1_22.1.2205-1_level_0_66.hdf5#
# home_ainur-karimov_data_raw_29.01.23_Датасеты биопсии C16_13323 (22.1.3015)_13323 (22.1.3015)_level_0_1484.hdf5
# /mnt/datastore/WRITE_ACCESS_DIR/patho/cropped/stomach/1/256/train/29.01.23_Датасеты биопсии C16_1002931_1002931_level_0_15.hdf5

# щас данные новые и вот такие в трейне есть
# raw_24_data_кишечник_100062-5_level_0_5
# raw_24_data_кишечник_100541-5_level_0_68

# сделать мена папок полные например 04.07.22_Размеченные_22.2.4739
# а файлы level_0_78.hdf5

# 950 стекл всего примерно или еще 250


# home_ainur-karimov_data_raw_
# 04.07.22_
# Размеченные_
# 22.2.4739_
# level_0_22.hdf5


# это нормальный код, работает

print("ok")
