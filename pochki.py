import SimpleITK as sitk
import nrrd
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from scipy.ndimage import zoom
import SimpleITK as sitk


# а здесь 2
# def extract_class_names(segmentation_file):
#     # Чтение файла сегментации
#     seg = sitk.ReadImage(segmentation_file)
#
#     # Получение метаинформации
#     metadata_dict = {}
#     for k in seg.GetMetaDataKeys():
#         metadata_dict[k] = seg.GetMetaData(k)
#
#     # Извлечение информации о сегментах
#     class_names = []
#     i = 0
#     while True:
#         segment_name_key = f"Segment{i}_Name"
#         if segment_name_key in metadata_dict:
#             segment_name = metadata_dict[segment_name_key]
#             if not segment_name.startswith("Segment_"):
#                 class_names.append(segment_name)
#             i += 1
#         else:
#             break
#
#     return class_names


# вот тут много классов


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
# class_names = extract_class_names(file)
# print("class_names", class_names)


def read_nrrd_and_save_all_slices(path):
    data, header = nrrd.read(path)
    print("header", header)
    print(np.max(data))
    cmap = plt.cm.get_cmap(
        "tab20", np.max(data) + 1
    )  # Создаем колор мапу с нужным количеством цветов
    output_dir = "/home/imran/Документы/Innopolis/First_data_test/output_nrrd_json"
    os.makedirs(output_dir, exist_ok=True)

    len = data.shape[2]
    print("data.shape", data.shape)
    print("len", len)

    for i in range(len):
        d = data[:, :, i]

        # unique_values, counts = np.unique(d, return_counts=True)
        # print(f"Slice {i+1} unique values and their counts in mask:")
        # for value, count in zip(unique_values, counts):
        #     print(f"Value: {value}, Count: {count}")

        if d.max() != 0:
            # unique_values, counts = np.unique(d, return_counts=True)
            # print(f"Slice {i+1} unique values and their counts in mask:")
            # for value, count in zip(unique_values, counts):
            #     print(f"Value: {value}, Count: {count}")

            unique_values, indices = np.unique(d, return_inverse=True)
            plt.imshow(d, cmap=cmap, interpolation="nearest")
            plt.colorbar(
                ticks=range(np.max(data) + 1)
            )  # Показываем colorbar для наглядности
            file_path = os.path.join(output_dir, f"slice_{i+1}.png")
            plt.savefig(file_path)
            plt.close()
            print(f"Slice {i+1} saved as {file_path}.")


#
# def read_nrrd_and_save_all_slices_cv2(image_path, mask_path):
#     # Чтение основного изображения и маски
#     data, header = nrrd.read(image_path)
#     mask_data, mask_header = nrrd.read(mask_path)
#
#     print("Максимальное значение в данных:", np.max(data))
#     print("Максимальное значение в маске:", np.max(mask_data))
#
#     output_dir = "nrrd_output_images_cv2"
#     os.makedirs(output_dir, exist_ok=True)
#
#     for i in range(data.shape[2]):
#         d = data[:, :, i]
#         m = mask_data[:, :, i]
#
#         # print("m", m)
#         unique_values, counts = np.unique(m, return_counts=True)
#         print(f"Slice {i+1} unique values and their counts in mask:")
#         for value, count in zip(unique_values, counts):
#             print(f"Value: {value}, Count: {count}")
#
#         if d.max() != 0:
#             # Нормализуем данные для отображения в формате uint8 (0-255)
#             normalized_img = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(
#                 np.uint8
#             )
#             # Преобразуем в цветное изображение (gray)
#             gray_img = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2BGR)
#             print("gray shape", gray_img.shape)
#
#             # Создаем цветное изображение для контуров
#             color_contours_img = np.zeros_like(gray_img)
#
#             print("color_contours_img shape", color_contours_img.shape)
#
#             # Найти контуры маски
#             contours, _ = cv2.findContours(
#                 m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#             )
#             print("contours", contours)
#
#             # Нарисовать контуры на цветном изображении
#             cv2.drawContours(
#                 color_contours_img, contours, -1, (0, 255, 0), 1
#             )  # Зеленые контуры
#
#             # Объединяем серое изображение и цветные контуры
#             combined_img = cv2.addWeighted(gray_img, 1, color_contours_img, 0.5, 0)
#
#             # Сохраняем изображение
#             file_path = os.path.join(output_dir, f"slice_{i+1}.png")
#             cv2.imwrite(file_path, combined_img)
#             print(f"Slice {i+1} saved as {file_path}.")
#


# это не очень интерполирует и по папкам не раскидывает
# def read_nrrd_and_save_all_slices_cv2(image_dir, mask_path):
#     # Чтение маски
#     mask_data, mask_header = nrrd.read(mask_path)
#
#     # Считываем все файлы изображений из директории
#     for file_name in os.listdir(image_dir):
#         if file_name.endswith(".nrrd"):
#             image_path = os.path.join(image_dir, file_name)
#
#             # Чтение основного изображения
#             data, header = nrrd.read(image_path)
#
#             # Интерполяция маски до размера изображения
#             interpolated_mask = np.zeros(
#                 (mask_data.shape[0], data.shape[0], data.shape[1], data.shape[2])
#             )
#             for i in range(mask_data.shape[0]):
#                 zoom_factors = (
#                     data.shape[0] / mask_data.shape[1],
#                     data.shape[1] / mask_data.shape[2],
#                     data.shape[2] / mask_data.shape[3],
#                 )
#                 interpolated_mask[i] = zoom(mask_data[i], zoom_factors, order=0)
#
#             print("Максимальное значение в данных:", np.max(data))
#             print("Максимальное значение в маске:", np.max(interpolated_mask))
#
#             output_dir = "nrrd_output_images_cv2"
#             os.makedirs(output_dir, exist_ok=True)
#
#             for i in range(data.shape[2]):
#                 d = data[:, :, i]
#
#                 # Применение каждого слоя маски
#                 for j in range(interpolated_mask.shape[0]):
#                     m = interpolated_mask[j, :, :, i]
#
#                     unique_values, counts = np.unique(m, return_counts=True)
#                     print(
#                         f"Slice {i+1}, Layer {j+1} unique values and their counts in mask:"
#                     )
#                     for value, count in zip(unique_values, counts):
#                         print(f"Value: {value}, Count: {count}")
#
#                     if d.max() != 0:
#                         # Нормализуем данные для отображения в формате uint8 (0-255)
#                         normalized_img = cv2.normalize(
#                             d, None, 0, 255, cv2.NORM_MINMAX
#                         ).astype(np.uint8)
#                         # Преобразуем в цветное изображение (gray)
#                         gray_img = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2BGR)
#
#                         # Создаем цветное изображение для контуров
#                         color_contours_img = np.zeros_like(gray_img)
#
#                         # Найти контуры маски
#                         contours, _ = cv2.findContours(
#                             m.astype(np.uint8),
#                             cv2.RETR_EXTERNAL,
#                             cv2.CHAIN_APPROX_SIMPLE,
#                         )
#
#                         # Нарисовать контуры на цветном изображении
#                         cv2.drawContours(
#                             color_contours_img, contours, -1, (0, 255, 0), 1
#                         )  # Зеленые контуры
#
#                         # Объединяем серое изображение и цветные контуры
#                         combined_img = cv2.addWeighted(
#                             gray_img, 1, color_contours_img, 0.5, 0
#                         )
#
#                         # Сохраняем изображение
#                         file_path = os.path.join(
#                             output_dir,
#                             f"{os.path.splitext(file_name)[0]}_slice_{i+1}_layer_{j+1}.png",
#                         )
#                         cv2.imwrite(file_path, combined_img)
#                         print(f"Slice {i+1}, Layer {j+1} saved as {file_path}.")
#


def resample_volume(image, reference_image, interpolator=sitk.sitkLinear):
    reference_size = reference_image.GetSize()
    reference_spacing = reference_image.GetSpacing()
    reference_direction = reference_image.GetDirection()
    reference_origin = reference_image.GetOrigin()

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(reference_spacing)
    resample.SetSize(reference_size)
    resample.SetOutputDirection(reference_direction)
    resample.SetOutputOrigin(reference_origin)
    resample.SetInterpolator(interpolator)

    resampled_image = resample.Execute(image)
    return resampled_image


def read_nrrd_and_save_all_slices_cv2(image_dir, mask_path):
    # Чтение маски
    mask_data, mask_header = nrrd.read(mask_path)
    mask = sitk.GetImageFromArray(mask_data)

    # Считываем все файлы изображений из директории
    for file_name in os.listdir(image_dir):
        print("file_name", file_name)
        if file_name.endswith(".nrrd"):
            image_path = os.path.join(image_dir, file_name)

            # Чтение основного изображения
            data, header = nrrd.read(image_path)
            image = sitk.GetImageFromArray(data)

            # Ресэмплирование маски до размера изображения
            resampled_mask = resample_volume(mask, image, sitk.sitkNearestNeighbor)
            resampled_mask_data = sitk.GetArrayFromImage(resampled_mask)

            print("Максимальное значение в данных:", np.max(data))
            print("Максимальное значение в маске:", np.max(resampled_mask_data))

            for layer in range(resampled_mask_data.shape[0]):
                output_dir = f"/home/imran/Документы/Innopolis/First_data_test/nrrd_output_images_cv2/layer_{layer+1}"
                os.makedirs(output_dir, exist_ok=True)

                for i in range(data.shape[2]):
                    d = data[:, :, i]
                    m = resampled_mask_data[layer, :, :, i]

                    unique_values, counts = np.unique(m, return_counts=True)
                    print(
                        f"Slice {i+1}, Layer {layer+1} unique values and their counts in mask:"
                    )
                    for value, count in zip(unique_values, counts):
                        print(f"Value: {value}, Count: {count}")

                    if d.max() != 0:
                        # Нормализуем данные для отображения в формате uint8 (0-255)
                        normalized_img = cv2.normalize(
                            d, None, 0, 255, cv2.NORM_MINMAX
                        ).astype(np.uint8)
                        # Преобразуем в цветное изображение (gray)
                        gray_img = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2BGR)

                        # Создаем цветное изображение для контуров
                        color_contours_img = np.zeros_like(gray_img)

                        # Определяем цвета для каждого класса
                        class_colors = {
                            1: (0, 255, 0),  # Зеленый для класса 1
                            2: (0, 0, 255),  # Красный для класса 2
                            3: (255, 0, 0),  # Синий для класса 3
                            4: (0, 255, 255),  # Желтый для класса 4
                            5: (255, 0, 255),  # Фиолетовый для класса 5
                            6: (255, 255, 0),  # Циан для класса 6
                        }

                        # Нарисовать контуры для каждого класса
                        for class_value, color in class_colors.items():
                            class_mask = (m == class_value).astype(np.uint8)
                            contours, _ = cv2.findContours(
                                class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                            )
                            cv2.drawContours(color_contours_img, contours, -1, color, 1)

                        # Объединяем серое изображение и цветные контуры
                        combined_img = cv2.addWeighted(
                            gray_img, 1, color_contours_img, 0.5, 0
                        )

                        # Сохраняем изображение
                        file_path = os.path.join(
                            output_dir,
                            f"{os.path.splitext(file_name)[0]}_slice_{i+1}.png",
                        )
                        cv2.imwrite(file_path, combined_img)
                        print(f"Slice {i+1}, Layer {layer+1} saved as {file_path}.")


# Директория с изображениями и путь к файлу маски
image_dir = "/home/imran/Документы/Innopolis/First_data_test/Проект _Почки_/Проект Почки/готовые 1 часть/Sheremetjev MJu/3D"
mask_path = image_dir + "/" + "Segmentation_1.seg.nrrd"

# Запуск функции
read_nrrd_and_save_all_slices_cv2(image_dir, mask_path)
