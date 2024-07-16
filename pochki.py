import nrrd
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2


def read_nrrd_and_save_all_slices(path):
    data, header = nrrd.read(path)
    print(np.max(data))
    cmap = plt.cm.get_cmap(
        "tab20", np.max(data) + 1
    )  # Создаем колор мапу с нужным количеством цветов
    output_dir = "nrrd_output_images"
    os.makedirs(output_dir, exist_ok=True)

    len = data.shape[2]
    print("len", len)

    for i in range(len):
        d = data[:, :, i]

        # unique_values, counts = np.unique(d, return_counts=True)
        # print(f"Slice {i+1} unique values and their counts in mask:")
        # for value, count in zip(unique_values, counts):
        #     print(f"Value: {value}, Count: {count}")

        if d.max() != 0:
            unique_values, counts = np.unique(d, return_counts=True)
            print(f"Slice {i+1} unique values and their counts in mask:")
            for value, count in zip(unique_values, counts):
                print(f"Value: {value}, Count: {count}")

            unique_values, indices = np.unique(d, return_inverse=True)
            plt.imshow(d, cmap=cmap, interpolation="nearest")
            plt.colorbar(
                ticks=range(np.max(data) + 1)
            )  # Показываем colorbar для наглядности
            file_path = os.path.join(output_dir, f"slice_{i+1}.png")
            plt.savefig(file_path)
            plt.close()
            print(f"Slice {i+1} saved as {file_path}.")


read_nrrd_and_save_all_slices(
    "/home/imran/Документы/Innopolis/First_data_test/Elizarova NV/АФ/Segmentation.seg.nrrd"
)

# read_nrrd_and_save_all_slices(
#     "/home/imran/Документы/Innopolis/First_data_test/Elizarova NV/dicom/scene/Data/Segmentation.seg.nrrd"
# )
# 3 Abdomen  1.0  B30f.nrrd изображение
# Segmentation.seg.nrrd len 678 маска
# Segmentation preview.seg.nrrd len 159
# 3 Abdomen  1.0  B30f.nrrd len 678


# def read_nrrd_and_save_all_slices_cv2(path):
#     data, header = nrrd.read(path)
#     print(np.max(data))
#     output_dir = "nrrd_output_images_cv2"
#     os.makedirs(output_dir, exist_ok=True)
#
#     max_value = np.max(data)
#     font = cv2.FONT_HERSHEY_COMPLEX
#
#     for i in range(data.shape[2]):
#         d = data[:, :, i]
#         if d.max() != 0:
#             # Нормализуем данные для отображения в формате uint8 (0-255)
#             normalized_img = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(
#                 np.uint8
#             )
#             # Преобразуем в цветное изображение
#             color_img = cv2.applyColorMap(normalized_img, cv2.COLORMAP_JET)
#
#             # Создаем цветовую шкалу
#             color_bar = np.linspace(0, max_value, num=256, endpoint=True).astype(
#                 np.uint8
#             )
#             color_bar_img = cv2.applyColorMap(color_bar, cv2.COLORMAP_JET)
#             color_bar_img = cv2.resize(
#                 color_bar_img, (50, color_img.shape[0]), interpolation=cv2.INTER_NEAREST
#             )
#
#             # Создаем окончательное изображение с добавленной цветовой шкалой
#             final_img = np.zeros(
#                 (color_img.shape[0], color_img.shape[1] + color_bar_img.shape[1], 3),
#                 dtype=np.uint8,
#             )
#             final_img[:, : color_img.shape[1]] = color_img
#             final_img[:, color_img.shape[1] :] = color_bar_img
#
#             # Сохраняем изображение
#             file_path = os.path.join(output_dir, f"slice_{i+1}.png")
#             cv2.imwrite(file_path, final_img)
#             print(f"Slice {i+1} saved as {file_path}.")


def read_nrrd_and_save_all_slices_cv2(image_path, mask_path):
    # Чтение основного изображения и маски
    data, header = nrrd.read(image_path)
    mask_data, mask_header = nrrd.read(mask_path)

    print("Максимальное значение в данных:", np.max(data))
    print("Максимальное значение в маске:", np.max(mask_data))

    output_dir = "nrrd_output_images_cv2"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(data.shape[2]):
        d = data[:, :, i]
        m = mask_data[:, :, i]

        # print("m", m)
        unique_values, counts = np.unique(m, return_counts=True)
        print(f"Slice {i+1} unique values and their counts in mask:")
        for value, count in zip(unique_values, counts):
            print(f"Value: {value}, Count: {count}")

        if d.max() != 0:
            # Нормализуем данные для отображения в формате uint8 (0-255)
            normalized_img = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )
            # Преобразуем в цветное изображение (gray)
            gray_img = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2BGR)
            print("gray shape", gray_img.shape)

            # Создаем цветное изображение для контуров
            color_contours_img = np.zeros_like(gray_img)

            print("color_contours_img shape", color_contours_img.shape)

            # Найти контуры маски
            contours, _ = cv2.findContours(
                m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            print("contours", contours)

            # Нарисовать контуры на цветном изображении
            cv2.drawContours(
                color_contours_img, contours, -1, (0, 255, 0), 1
            )  # Зеленые контуры

            # Объединяем серое изображение и цветные контуры
            combined_img = cv2.addWeighted(gray_img, 1, color_contours_img, 0.5, 0)

            # Сохраняем изображение
            file_path = os.path.join(output_dir, f"slice_{i+1}.png")
            cv2.imwrite(file_path, combined_img)
            print(f"Slice {i+1} saved as {file_path}.")


# read_nrrd_and_save_all_slices_cv2(
#     "/home/imran/Документы/Innopolis/First_data_test/Elizarova NV/dicom/scene/Data/3 Abdomen  1.0  B30f.nrrd",
#     "/home/imran/Документы/Innopolis/First_data_test/Elizarova NV/dicom/scene/Data/Segmentation.seg.nrrd",
# )
