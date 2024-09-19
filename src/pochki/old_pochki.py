# это вроде чтобы конкретный файл маски отрисовать, а в pochki.py он сам найдет все,
# но на самом деле я не помню уже что это за файлы, потому что есть файл data_convert.py


import SimpleITK as sitk
import nrrd
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from scipy.ndimage import zoom
import SimpleITK as sitk


def extract_class_names(segmentation_file):
    # Чтение файла сегментации
    data, header = nrrd.read(segmentation_file)

    # Извлечение информации о сегментах
    class_names = []
    i = 0
    while True:
        segment_name_key = f"Segment{i}_Name"
        if segment_name_key in header:
            segment_name = header[segment_name_key]
            # Добавление всех имен сегментов
            if segment_name:
                class_names.append(segment_name)
            i += 1
        else:
            break

    return class_names


file = "/home/imran/Документы/Innopolis/First_data_test/Проект _Почки_/Проект Почки/готовые 1 часть/Sheremetjev MJu/3D/Segmentation_1.seg.nrrd"
class_names = extract_class_names(file)
print("class_names", class_names)
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
    interpolator = sitk.sitkLinear
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


def read_nrrd_and_save_all_slices_cv2(image_dir, mask_dir):
    # Чтение масок и изображений
    for file_name in os.listdir(image_dir):
        if file_name.endswith(".nrrd") and "Segmentation" not in file_name:
            image_path = os.path.join(image_dir, file_name)
            print("image_path", image_path)
            reference_image = sitk.ReadImage(image_path)

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
            print(
                "type resampled_mask", type(resampled_mask)
            )  # <class 'numpy.ndarray'>
            # resampled_mask_data = sitk.GetArrayFromImage(resampled_mask)
            resampled_mask_data = resampled_mask

            np_mask = sitk.GetArrayFromImage(mask)
            np_data = sitk.GetArrayFromImage(reference_image)
            print("np_mask shape", np_mask.shape)
            print("np_data shape", np_data.shape)
            print("resampled_mask shape", np.shape(resampled_mask_data))

            # по идее надо вот их сравнить, то есть до и после,
            # layer_0 = np_mask[23, :, :, 0]
            # layer_1 = np_mask[23, :, :, 1]

            #  но все работает норм и можно break и эти принты убрать
            # layer_0 = resampled_mask_data[23, :, :, 0]
            # layer_1 = resampled_mask_data[23, :, :, 1]

            # # Получение уникальных значений и их количества для слоя 0
            # unique_values_layer_0, counts_layer_0 = np.unique(layer_0, return_counts=True)
            # print("Уникальные значения и их количество в resampled_mask_data[23, :, :, 0]:")
            # for value, count in zip(unique_values_layer_0, counts_layer_0):
            #     print(f"Value: {value}, Count: {count}")

            # # Получение уникальных значений и их количества для слоя 1
            # unique_values_layer_1, counts_layer_1 = np.unique(layer_1, return_counts=True)
            # print("Уникальные значения и их количество в resampled_mask_data[23, :, :, 1]:")
            # for value, count in zip(unique_values_layer_1, counts_layer_1):
            #     print(f"Value: {value}, Count: {count}")

            # break

            for layer in range(resampled_mask_data.shape[-1]):
                output_dir = f"/home/alexskv/pochki/nrrd_output_images_cv2/{os.path.splitext(file_name)[0]}_layer_{layer+1}"
                os.makedirs(output_dir, exist_ok=True)

                for i in range(np_data.shape[0]):
                    d = np_data[i, :, :]
                    m = resampled_mask_data[i, :, :, layer]

                    unique_values, counts = np.unique(m, return_counts=True)
                    print(
                        f"Slice {i+1}, Layer {layer+1} unique values and their counts in mask:"
                    )
                    for value, count in zip(unique_values, counts):
                        print(f"Value: {value}, Count: {count}")

                    if d.max() != 0:
                        normalized_img = cv2.normalize(
                            d, None, 0, 255, cv2.NORM_MINMAX
                        ).astype(np.uint8)
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
                            contours, _ = cv2.findContours(
                                class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                            )
                            cv2.drawContours(color_contours_img, contours, -1, color, 1)

                        combined_img = cv2.addWeighted(
                            gray_img, 1, color_contours_img, 0.5, 0
                        )
                        file_path = os.path.join(
                            output_dir,
                            f"{os.path.splitext(file_name)[0]}_slice_{i+1}.png",
                        )
                        cv2.imwrite(file_path, combined_img)
                        print(f"Slice {i+1}, Layer {layer+1} saved as {file_path}.")


# # Директория с изображениями и путь к файлу маски
# image_dir = "/home/alexskv/data/Проект_Почки/готовые_1_часть/Sheremetjev MJu/3D"
# mask_path = image_dir + "/" + "Segmentation_1.seg.nrrd"

# read_nrrd_and_save_all_slices_cv2(image_dir, image_dir)


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
