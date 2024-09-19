import SimpleITK as sitk
import nrrd
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from scipy.ndimage import zoom
import SimpleITK as sitk
from collections import OrderedDict


def old_extract_class_names(segmentation_file):
    # Чтение файла сегментации
    data, header = nrrd.read(segmentation_file)

    # Извлечение информации о сегментах
    class_names = []
    i = 0
    while True:
        segment_name_key = f"Segment{i}_Name"
        print("header", header)
        if segment_name_key in header:
            segment_name = header[segment_name_key]
            # Добавление всех имен сегментов
            if segment_name:
                class_names.append(segment_name)
            i += 1
        else:
            break

    return class_names


# image_dir = "/home/alexskv/data/Проект_Почки/готовые_1_часть/Sheremetjev MJu/3D"
# mask_path = image_dir + "/" + "Segmentation_1.seg.nrrd"

# file = "/home/imran/Документы/Innopolis/First_data_test/Проект _Почки_/Проект Почки/готовые 1 часть/Sheremetjev MJu/3D/Segmentation_1.seg.nrrd"
# class_names = extract_class_names(mask_path)
# print("class_names", class_names)

# вроде нашлись все классы
# class_names[
#     "right_kidney_ID1",
#     "right_kidney_upper_segment_ID2",
#     "right_kidney_middle_segment_ID3",
#     "right_kidney_lower_segment_ID4",
#     "left_kidney_ID5",
#     "left_kidney_upper_segment_ID6",
#     "left_kidney_middle_segment_ID7",
#     "left_kidney_lower_segment_ID8",
#     "malignant_tumor_ID9",
#     "benign_tumor_ID10",
#     "cyst_ID11",
#     "abscess_ID12",
# ]
#




def resample_volume(mask, reference_image):
    
    interpolator=sitk.sitkLinear
    reference_size = reference_image.GetSize()
    print(reference_size, "reference_size")
    reference_spacing = reference_image.GetSpacing()
    print(reference_spacing, "reference_spacing")
    reference_direction = reference_image.GetDirection()
    print(reference_direction, "reference_direction")
    reference_origin = reference_image.GetOrigin()
    print(reference_origin, "reference_origin")

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(reference_spacing)
    resample.SetSize(reference_size)
    resample.SetOutputDirection(reference_direction)
    resample.SetOutputOrigin(reference_origin)
    resample.SetInterpolator(interpolator)
    
    resampled_mask = resample.Execute(mask)
    return sitk.GetArrayFromImage(resampled_mask)

def calculate_window_parameters(image):
    # Преобразование изображения в массив и вычисление статистик
    image_array = sitk.GetArrayFromImage(image)
    mean_intensity = np.mean(image_array)
    std_intensity = np.std(image_array)

    # Расчет window level и window width
    window_level = mean_intensity
    window_width = 2 * std_intensity
    return window_level, window_width

def adjust_window_level(image, window_level, window_width):
    # Расчет минимального и максимального значения для окна
    min_window = window_level - (window_width / 2)
    max_window = window_level + (window_width / 2)
    
    # Применение оконирования
    return sitk.IntensityWindowing(image, windowMinimum=min_window, windowMaximum=max_window, 
                                   outputMinimum=0.0, outputMaximum=255.0)



def extract_class_names(segmentation_file):
    # Чтение файла сегментации
    data, header = nrrd.read(segmentation_file)
    # print("header", header)
    print("data shape", data.shape)
    values, counts = np.unique(data, return_counts=True)
    print("Уникальные значения и их количество в data")
    for value, count in zip(values, counts):
        print(f"Value: {value}, Count: {count}")
    
    mask = sitk.ReadImage(mask_path)

    # Извлечение информации о сегментах
    class_names = OrderedDict()
    i = 0
    while True:
        segment_name_key = f"Segment{i}_Name"
        if segment_name_key in header:
            segment_name = header[segment_name_key]
            segment_layer = header[f"Segment{i}_Layer"]
            segment_label_value = header[f"Segment{i}_LabelValue"]
            class_names[(int(segment_layer), int(segment_label_value))] = segment_name
            i += 1
        else:
            break

    return class_names, mask #data


def convert_to_one_hot(np_mask, class_names):
    # Получение уникальных значений классов из маски
    unique_classes = np.unique(np_mask)
    unique_classes = unique_classes[unique_classes != 0]  # Исключаем фон (значение 0)
    num_classes = len(unique_classes)
    
        
    
    one_hot_shape = np_mask.shape[:3] + 12 #(num_classes,) 
    one_hot_mask = np.zeros(one_hot_shape, dtype=np.uint8)
    
    # Создание словаря индексов классов
    class_indices = {value: idx for idx, value in enumerate(unique_classes)}
    print("class_indices", class_indices) 

    # Создание словаря для отображения индекса слоя на имя класса
    class_layer_to_name = {}

    for (layer, label_value), class_name in class_names.items():
        if label_value in class_indices:
            unique_class_idx = class_indices[label_value]
            one_hot_mask[..., unique_class_idx] = (np_mask[..., layer] == label_value).astype(np.uint8)
            class_layer_to_name[unique_class_idx] = class_name
    
    return one_hot_mask, class_indices, class_layer_to_name


def read_nrrd_and_save_all_slices_cv2(image_dir, mask_dir):
    for file_name in os.listdir(image_dir):
        if file_name.endswith(".nrrd") and "Segmentation" not in file_name:
            image_path = os.path.join(image_dir, file_name)
            print("image_path", image_path)
            reference_image = sitk.ReadImage(image_path)

            level, width = calculate_window_parameters(reference_image)
            adjusted_image = adjust_window_level(reference_image, level, width)

            mask_file_name = "Segmentation_1.seg.nrrd"
            mask_path = os.path.join(mask_dir, mask_file_name)
            if not os.path.exists(mask_path):
                print(f"Mask not found for {file_name}, skipping...")
                continue

            print("mask_path", mask_path)
            # class_names, np_mask = extract_class_names(mask_path)
            class_names, mask = extract_class_names(mask_path)
            resampled_mask = resample_volume(mask, reference_image)
            print("resampled_mask shape", resampled_mask.shape)
            np_mask = sitk.GetArrayFromImage(mask)
            
            print("class_names", class_names)
             # Вывод уникальных значений для всех слоев маски
            for layer in range(np_mask.shape[-1]):
                unique_values_layer, counts_layer = np.unique(np_mask[..., layer], return_counts=True)
                print(f"Уникальные значения и их количество в np_mask[..., {layer}]:")
                for value, count in zip(unique_values_layer, counts_layer):
                    print(f"Value: {value}, Count: {count}")

                

            # resampled_mask = resample_volume(sitk.GetImageFromArray(np_mask), reference_image)
            resampled_mask_data = resampled_mask
        
            
            np_data = sitk.GetArrayFromImage(adjusted_image)
            # print("np_data", np_data)
            print("np_mask shape", np_mask.shape) # (172, 512, 512, 2)
            print("np_data shape", np_data.shape)
            print("resampled_mask shape", resampled_mask_data.shape)

    
            # one_hot_mask = convert_to_one_hot(resampled_mask_data, class_names)
            one_hot_mask, class_indices, class_layer_to_name = convert_to_one_hot(resampled_mask_data, class_names)
            print("one_hot_mask shape", one_hot_mask.shape) # one_hot_mask shape (117, 512, 512, 7)
            print("class_layer_to_name", class_layer_to_name)
            
            
            break
            # for i in range(one_hot_mask.shape[0]):
            #     d = np_data[i, :, :]
            #     m = one_hot_mask[i, :, :, :]

            #     unique_values, counts = np.unique(m, return_counts=True)
            #     print(f"Slice {i+1} unique values and their counts in mask:")
            #     for value, count in zip(unique_values, counts):
            #         print(f"Value: {value}, Count: {count}")

            #     if d.max() != 0 and m.max() != 0:
            #         normalized_img = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            #         gray_img = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2BGR)
            #         color_contours_img = np.zeros_like(gray_img)

            #         class_colors = {
            #             0: (0, 255, 0),
            #             1: (0, 0, 255),
            #             2: (255, 0, 0),
            #             3: (0, 255, 255),
            #             4: (255, 0, 255),
            #             5: (255, 255, 0),
            #             6: (128, 0, 128),
            #             7: (128, 128, 0),
            #             8: (0, 128, 128),
            #             9: (128, 128, 128),
            #             10: (0, 0, 128),
            #             11: (0, 128, 0)
            #         }

            #         for layer, color in class_colors.items():
            #             if layer >= m.shape[-1]:
            #                 continue
            #             class_mask = (m[..., layer] == 1).astype(np.uint8)
            #             contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #             cv2.drawContours(color_contours_img, contours, -1, color, 1)
            #             # print("contours", contours)

            #         combined_img = cv2.addWeighted(gray_img, 1, color_contours_img, 0.5, 0)
            #         output_dir = f"/home/alexskv/pochki/nrrd_output_images_cv2/{os.path.splitext(file_name)[0]}_layer_{layer+1}"
            #         os.makedirs(output_dir, exist_ok=True)
            #         file_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_slice_{i+1}.png")
            #         cv2.imwrite(file_path, combined_img)
            #         print(f"Slice {i+1} saved as {file_path}.")

            output_base_dir = f"/home/alexskv/pochki/nrrd_output_images_cv2/{os.path.splitext(file_name)[0]}"
            for class_idx in range(one_hot_mask.shape[-1]):
                class_dir = os.path.join(output_base_dir, f"class_{class_idx}")
                os.makedirs(class_dir, exist_ok=True)
            
            
            for i in range(one_hot_mask.shape[0]):
                d = sitk.GetArrayViewFromImage(adjusted_image)[i, :, :]
                m = one_hot_mask[i, :, :, :]

                unique_values, counts = np.unique(m, return_counts=True)
                print(f"Slice {i+1} unique values and their counts in mask:")
                for value, count in zip(unique_values, counts):
                    print(f"Value: {value}, Count: {count}")

                if d.max() != 0 and m.max() != 0:
                    normalized_img = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    gray_img = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2BGR)
                    
                    for class_idx in range(one_hot_mask.shape[-1]):
                        color_contours_img = np.zeros_like(gray_img)
                        class_mask = (m[..., class_idx] == 1).astype(np.uint8)
                        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        print("contours", contours)
                        cv2.drawContours(color_contours_img, contours, -1, (0, 255, 0), 1)

                        combined_img = cv2.addWeighted(gray_img, 1, color_contours_img, 0.5, 0)
                        class_dir = os.path.join(output_base_dir, f"class_{class_idx}")
                        file_path = os.path.join(class_dir, f"{os.path.splitext(file_name)[0]}_slice_{i+1}.png")
                        cv2.imwrite(file_path, combined_img)
                        print(f"Slice {i+1} for class {class_idx} saved as {file_path}.")



def old_read_nrrd_and_save_all_slices_cv2(image_dir, mask_dir):
    # Чтение масок и изображений
    for file_name in os.listdir(image_dir):
        if file_name.endswith(".nrrd") and "Segmentation" not in file_name:
            image_path = os.path.join(image_dir, file_name)
            print("image_path", image_path)
            reference_image = sitk.ReadImage(image_path)

            level, width = calculate_window_parameters(reference_image)
            adjusted_image = adjust_window_level(reference_image, level, width)

            # Определение пути к соответствующей маске
            mask_file_name = "Segmentation_1.seg.nrrd"
            mask_path = os.path.join(mask_dir, mask_file_name)
            if not os.path.exists(mask_path):
                print(f"Mask not found for {file_name}, skipping...")
                continue

            print("mask_path", mask_path)
            mask = sitk.ReadImage(mask_path)

            print("resample")
            resampled_mask = resample_volume(mask, reference_image)
            print("resampled_mask shape", np.shape(resampled_mask))
            print("type resampled_mask", type(resampled_mask)) # <class 'numpy.ndarray'>
            # resampled_mask_data = sitk.GetArrayFromImage(resampled_mask)
            resampled_mask_data = resampled_mask
            
            np_mask = sitk.GetArrayFromImage(mask)
            # np_data = sitk.GetArrayFromImage(reference_image)
            np_data = sitk.GetArrayFromImage(adjusted_image)
            # print("np_data", np_data)
            print("np_mask shape", np_mask.shape)
            print("np_data shape", np_data.shape)
            print("resampled_mask shape", np.shape(resampled_mask_data))
            break
            layer_0 = np_mask[23, :, :, 0]
            layer_1 = np_mask[23, :, :, 1]
            
            # по идее надо вот их сравнить, то есть до и после
            layer_0 = resampled_mask_data[23, :, :, 0]
            layer_1 = resampled_mask_data[23, :, :, 1]

            # Получение уникальных значений и их количества для слоя 1
            unique_values_layer_1, counts_layer_1 = np.unique(layer_1, return_counts=True)
            print("Уникальные значения и их количество в resampled_mask_data[23, :, :, 1]:")
            for value, count in zip(unique_values_layer_1, counts_layer_1):
                print(f"Value: {value}, Count: {count}")
                
        
            for layer in range(resampled_mask_data.shape[-1]):
                output_dir = f"/home/alexskv/pochki/nrrd_output_images_cv2/{os.path.splitext(file_name)[0]}_layer_{layer+1}"
                os.makedirs(output_dir, exist_ok=True)

                for i in range(np_data.shape[0]):
                    d = np_data[i, :, :]
                    m = resampled_mask_data[i, :, :, layer]

                    unique_values, counts = np.unique(m, return_counts=True)
                    print(f"Slice {i+1}, Layer {layer+1} unique values and their counts in mask:")
                    for value, count in zip(unique_values, counts):
                        print(f"Value: {value}, Count: {count}")

                    if d.max() != 0:
                        normalized_img = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        gray_img = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2BGR)
                        color_contours_img = np.zeros_like(gray_img)

                        class_colors = {
                            1: (0, 255, 0),  # Зеленый для класса 1
                            2: (0, 0, 255),  # Красный для класса 2
                            3: (255, 0, 0),  # Синий для класса 3
                            4: (0, 255, 255),  # Желтый для класса 4
                            5: (255, 0, 255),  # Фиолетовый для класса 5
                            6: (255, 255, 0),  # Циан для класса 6
                        }

                        for class_value, color in class_colors.items():
                            class_mask = (m == class_value).astype(np.uint8)
                            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(color_contours_img, contours, -1, color, 1)

                        combined_img = cv2.addWeighted(gray_img, 1, color_contours_img, 0.5, 0)
                        file_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_slice_{i+1}.png")
                        cv2.imwrite(file_path, combined_img)
                        print(f"Slice {i+1}, Layer {layer+1} saved as {file_path}.")






# Директория с изображениями и путь к файлу маски
image_dir = "/home/alexskv/data/Проект_Почки/готовые_часть_1/Sheremetjev MJu/3D"
mask_path = image_dir + "/" + "Segmentation_1.seg.nrrd"

read_nrrd_and_save_all_slices_cv2(image_dir, image_dir)





# image_dir = "/home/alexskv/data/Проект_Почки/готовые_1_часть/Sheremetjev MJu/3D"
# mask_path = image_dir + "/" + "Segmentation_1.seg.nrrd"

# # image_name = "2 Body Non Contrast 300 Br36 S3.nrrd"
# # image_name = "8 Body Arterial Phase 100 Br36 S3.nrrd"
# # image_name = "9 Body Venous Phase 300 Br36 S3.nrrd"
# image_name = "10 Body Delayed Phase 300 Br36 S3.nrrd"

# image_path = image_dir + "/" + image_name

# reference_image = sitk.ReadImage(image_path)
# mask = sitk.ReadImage(mask_path)
# np_mask = sitk.GetArrayFromImage(mask)
# np_image = sitk.GetArrayFromImage(reference_image)

# resampled_mask = resample_volume(mask, reference_image)

# print(np.shape(resampled_mask))
# print(np.shape(np_mask))
# print(np.shape(np_image))


