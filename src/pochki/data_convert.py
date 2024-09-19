import os
import json
import cv2
import h5py
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import SimpleITK as sitk
import nrrd
from to_coco import numpy_mask_to_coco_polygon, create_coco_annotation_from_mask

def old_resample_volume(mask, reference_image):
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


def resample_volume(mask, reference_image):
    interpolator = sitk.sitkNearestNeighbor  # Используем ближайшего соседа для интерполяции
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


def extract_class_names(segmentation_file): #надо изменить наверное 
    data, header = nrrd.read(segmentation_file)
    mask = sitk.ReadImage(segmentation_file)
    np_mask = sitk.GetArrayFromImage(mask)

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

    return class_names, np_mask


def convert_to_one_hot(np_mask, class_names, num_classes):
    # Создание one-hot маски
    num_classes = 12 #
    one_hot_shape = np_mask.shape[:3] + (num_classes,)
    one_hot_mask = np.zeros(one_hot_shape, dtype=np.uint8)

    # np_mask (242, 512, 512, 2)
    # one_hot_mask shape (242, 512, 512, 9)
    # Создание словаря индексов классов
    class_layer_to_name = {}
    class_idx = 0
    for (layer, label_value), class_name in class_names.items():
        # if class_idx < num_classes:
        one_hot_mask[..., class_idx] = (np_mask[..., layer] == label_value).astype(np.uint8)
        class_layer_to_name[class_idx] = class_name
        class_idx += 1

    # print("class_layer_to_name", class_layer_to_name)
    return one_hot_mask, class_layer_to_name


def find_segmentation_file(patient_dir):
    for file_name in os.listdir(patient_dir):
        if file_name.endswith(".seg.nrrd"):
            return os.path.join(patient_dir, file_name)
    return None


def apply_window_level(image, window_width, window_level):
    min_intensity = window_level - (window_width / 2)
    max_intensity = window_level + (window_width / 2)
    
    # Clip the image based on the window range
    image = np.clip(image, min_intensity, max_intensity)
    
    # Normalize the image to range [0, 255]
    image = ((image - min_intensity) / (max_intensity - min_intensity) * 255).astype(np.uint8)
    return image


def apply_clahe(image, clip_limit=4.0, tile_grid_size=(8, 8)):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        gray_image = image
    else:
        raise ValueError("Unexpected image format")

    gray_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_image = clahe.apply(gray_image)
    return enhanced_image


# Пример вызова функции
data_dir = "/home/alexskv/data/Проект_Почки"
output_json_path = "/home/alexskv/data/json_pochki/coco_annotations.json"
image_dir = "/home/alexskv/data/json_pochki"
# read_nrrd_and_save_all_slices_cv2(data_dir, output_json_path, image_dir)



# тут ресемпленная маска делается в one hot, а я хочу наоборот
def read_nrrd_and_save_all_slices_cv2(data_dir, output_base_dir):
    for part in os.listdir(data_dir):
        part_path = os.path.join(data_dir, part)
        if os.path.isdir(part_path):
            for patient_name in os.listdir(part_path):
                patient_dir = os.path.join(part_path, patient_name, "3D")
                print("patient_dir", patient_dir)
                if os.path.isdir(patient_dir):
                    for file_name in os.listdir(patient_dir):
                        file_path = os.path.join(patient_dir, file_name)
                        print("file_path", file_path)
                        if file_path.endswith(".nrrd") and "Segmentation" not in file_name:
                            image_path = file_path
                            reference_image = sitk.ReadImage(image_path) 
                            
                            # через nrrd пробую
                            # reference_image, header = nrrd.read(image_path)
                            # print("reference_image shape", reference_image.shape)
                            # cv2.imwrite("/home/alexskv/pochki/nrdd_out.jpg", reference_image[..., 0])
                            
                            #####                            
                            # level, width = calculate_window_parameters(reference_image, scale=1.5)
                            # reference_image = adjust_window_level(reference_image, level, width)
                            # reference_image = normalize_image(reference_image)
                            #####

                            mask_path = find_segmentation_file(patient_dir)
                            if not mask_path:
                                print(f"Mask not found for {file_name}, skipping...")
                                continue

                            class_names, nrrd_mask = extract_class_names(mask_path)
                            print("class_names", class_names)
                            mask = sitk.ReadImage(mask_path)
                            
                            resampled_mask = resample_volume(mask, reference_image)
                            unique_values_layer, counts_layer = np.unique(resampled_mask, return_counts=True)
                            print(f"Уникальные значения и их количество в resampled_mask:")
                            for value, count in zip(unique_values_layer, counts_layer):
                                print(f"Value: {value}, Count: {count}")
                                    
                            np_mask = sitk.GetArrayFromImage(mask)
                            print("np_mask shape", np_mask.shape) 
                            
                            all_unique_values = []
                            for layer in range(np_mask.shape[-1]):
                                unique_values_layer = np.unique(np_mask[..., layer])
                                print("unique_values_layer", unique_values_layer)
                                all_unique_values.extend(unique_values_layer)
                            
                            print("all_unique_values", all_unique_values)
                            all_unique_values = [x for x in all_unique_values if x != 0] # Исключаем фон (значение 0)
                            num_classes = len(all_unique_values) # + 1 # добавил для пустышек
                            print(f"Total unique classes: {num_classes}")
                            
                            for layer in range(np_mask.shape[-1]):
                                unique_values_layer, counts_layer = np.unique(np_mask[..., layer], return_counts=True)
                                print(f"Уникальные значения и их количество в np_mask[..., {layer}]:")
                                for value, count in zip(unique_values_layer, counts_layer):
                                    print(f"Value: {value}, Count: {count}")
                                    # print("type value", type(value))

                            one_hot_mask, class_layer_to_name = convert_to_one_hot(resampled_mask, class_names, num_classes)

                            print("class_layer_to_name", class_layer_to_name)
                            print("resampled_mask shape", resampled_mask.shape)
                            print("one_hot_mask shape", one_hot_mask.shape) # (242, 512, 512, 7)
                            np_data = sitk.GetArrayViewFromImage(reference_image)
                            print("np_data shape", np_data.shape) # (242, 512, 512)
                            # break
                            
                            output_coco_data = {
                                "info": {
                                    "version": "1.0",
                                    "year": 2024,
                                    "contributor": "",
                                    "description": "",
                                },
                                "licenses": [{"url": "", "id": 0, "name": ""}],
                                "images": [],
                                "annotations": [],
                                "categories": [],
                            }

                            for class_idx, class_name in class_layer_to_name.items():
                                output_coco_data["categories"].append({
                                    "id": class_idx + 1,
                                    "name": class_name,
                                    "supercategory": "",
                                })

                            patient_study_name = f"{patient_name}_{file_name[:-5]}"
                            output_patient_dir = os.path.join(output_base_dir, patient_study_name)
                            output_images_dir = os.path.join(output_patient_dir, "images")
                            output_annotations_dir = os.path.join(output_patient_dir, "annotations")
                            os.makedirs(output_images_dir, exist_ok=True)
                            os.makedirs(output_annotations_dir, exist_ok=True)

                            for i in range(one_hot_mask.shape[0]):
                                slice_image = sitk.GetArrayViewFromImage(reference_image)[i, :, :]
                                slice_mask = one_hot_mask[i, :, :, :]
                                
                                windowed_image = apply_window_level(slice_image, window_width=350, window_level=40)

                                enhanced_image = apply_clahe(windowed_image) # slice_image

                                # ниже slice_image было вместо enhanced_image
                                # result = cv2.normalize(enhanced_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                                result = windowed_image #enhanced_image # 
                                gray_img = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                                

                                image_id = f"{patient_name}_{file_name}_{i}"
                                out_convert_image = os.path.join(output_images_dir, f"{image_id}.jpg")
                                cv2.imwrite(out_convert_image, gray_img)
                                # print("out_convert_image", out_convert_image)
                                part = out_convert_image.split("/")[-1]
                                # print("part", part)
                                # break
                                output_coco_data["images"].append({
                                    "id": i, #image_id,
                                    "file_name": part, # out_convert_image,
                                    "width": gray_img.shape[1],
                                    "height": gray_img.shape[0],
                                })

                                for class_idx in range(slice_mask.shape[-1]):
                                    class_mask = (slice_mask[..., class_idx] == 1).astype(np.uint8)
                                    if np.sum(class_mask) > 0:
                                        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                        segmentation = [contour.flatten().tolist() for contour in contours if contour.size >= 6]
                                        if segmentation:
                                            annotation = {
                                                "id": len(output_coco_data["annotations"]) + 1, 
                                                "image_id": i, #image_id,
                                                "category_id": class_idx + 1,
                                                "segmentation": segmentation,
                                                "area": int(np.sum(class_mask)),
                                                "bbox": cv2.boundingRect(class_mask),
                                                "iscrowd": 0,
                                            }
                                            output_coco_data["annotations"].append(annotation)

                            output_json_path = os.path.join(output_annotations_dir, "instances_default.json")
                            with open(output_json_path, "w") as file:
                                json.dump(output_coco_data, file)






data_dir = "/home/alexskv/data/Проект_Почки"
output_base_dir = "/home/alexskv/data/json_pochki"
read_nrrd_and_save_all_slices_cv2(data_dir, output_base_dir)