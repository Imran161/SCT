# import os
# import json
# import pydicom
# import numpy as np
# from pathlib import Path
# import cv2

# def create_coco_annotations(patient_dir):
#     print("patient_dir", patient_dir)
#     images_dir = f"{patient_dir}/IMAGES"
#     masks_dir = f"{patient_dir}/npz_files"
    
#     part = str(patient_dir).split("/")[-1]
#     output_images_dir = Path(f"/home/imran-nasyrov/ainur_sct_json/{part}/images")
#     output_annotations_dir = Path(f"/home/imran-nasyrov/ainur_sct_json/{part}/annotations")
#     print("output_images_dir", output_images_dir)
#     print("output_annotations_dir", output_annotations_dir)

#     if not os.path.exists(output_images_dir):
#         os.makedirs(output_images_dir)
            
#     if not os.path.exists(output_annotations_dir):
#         os.makedirs(output_annotations_dir)

#     annotations = {
#         "images": [],
#         "annotations": [],
#         "categories": [
#             {"id": 1, "name": "hemorrhagic stroke", "supercategory": "object"},
#             {"id": 2, "name": "hemorrhagic stroke", "supercategory": "object"},
#             {"id": 3, "name": "hemorrhagic stroke", "supercategory": "object"},
#             {"id": 4, "name": "hemorrhagic stroke", "supercategory": "object"}
#         ]
#     }

#     # Load masks from NPZ file once
#     npz_file = Path(f"{masks_dir}/masks.npz")
#     if not npz_file.exists():
#         raise FileNotFoundError(f"NPZ file not found: {npz_file}")
    
#     mask_data = np.load(npz_file)
#     masks = mask_data.get('arr_0')
#     # print("masks", )
#     # values, counts = np.unique(masks, return_counts=True)
#     # print("values", values)
    
#     if masks is None:
#         raise ValueError(f"No masks found in NPZ file: {npz_file}")

#     num_slices, num_classes, height, width = masks.shape
#     print(f"Mask dimensions: {masks.shape}")

#     # new_height, new_width = 512, 512

#     # # Resize all masks to the new dimensions
#     # resized_masks = np.zeros((num_slices, num_classes, new_height, new_width), dtype=masks.dtype)
#     # for slice_index in range(num_slices):
#     #     for class_index in range(num_classes):
#     #         resized_masks[slice_index, class_index] = cv2.resize(masks[slice_index, class_index], (new_width, new_height))

#     annotation_id = 1
#     for image_id, dcm_file in enumerate(sorted(Path(images_dir).iterdir()), start=1):
#         if dcm_file.suffix.lower() != '.dcm':
#             continue

#         # Read the DICOM file
#         dicom_data = pydicom.dcmread(dcm_file)
        
#         # Define the file_name as the DICOM file name
#         file_name = dcm_file.name

#         # Collect image information
#         image_info = {
#             "id": image_id,
#             "file_name": file_name,
#             "width": width,
#             "height": height,
#         }
#         annotations["images"].append(image_info)

#         # Copy DICOM file to the new directory
#         output_image_path = output_images_dir / file_name
#         output_image_path.write_bytes(dcm_file.read_bytes())

#         # Ensure the index is within the range of the masks array
#         if image_id - 1 < num_slices:
#             for class_id in range(num_classes):
#                 mask = masks[image_id - 1, class_id, :, :]

#                 # Find contours using OpenCV
#                 contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
#                 segmentation = []
#                 for contour in contours:
#                     if len(contour) > 0:
#                         segmentation.append(contour.flatten().tolist())

#                 # Find bounding box
#                 y_indices, x_indices = np.where(mask > 0)
#                 if len(x_indices) == 0 or len(y_indices) == 0:
#                     continue

#                 x_min = int(x_indices.min())
#                 x_max = int(x_indices.max())
#                 y_min = int(y_indices.min())
#                 y_max = int(y_indices.max())

#                 bbox_width = x_max - x_min + 1
#                 bbox_height = y_max - y_min + 1

#                 # Calculate the area
#                 area = bbox_width * bbox_height

#                 # Create annotation for this mask
#                 annotation = {
#                     "id": annotation_id,
#                     "image_id": image_id,
#                     "category_id": class_id + 1,  # category_id should be from 1 to 4
#                     "bbox": [x_min, y_min, bbox_width, bbox_height],
#                     "area": area,
#                     "segmentation": segmentation,
#                     "iscrowd": 0
#                 }
#                 annotations["annotations"].append(annotation)
#                 annotation_id += 1

#     # Save annotations as JSON
#     annotations_file_path = output_annotations_dir / "instances_default.json"
#     with annotations_file_path.open("w") as f:
#         json.dump(annotations, f)#, indent=4)

# # Define the base directory where all patient data is stored
# base_dir = Path("/mnt/netstorage/Medicine/Medical/ainur_sct")

# # Iterate over each patient folder
# for patient_folder in base_dir.iterdir():
#     if patient_folder.is_dir():
#         create_coco_annotations(patient_folder)



import os
import json
import pydicom
import numpy as np
from pathlib import Path
import cv2

def create_coco_annotations(patient_dir):
    print("patient_dir", patient_dir)
    images_dir = f"{patient_dir}/IMAGES"
    masks_dir = f"{patient_dir}/npz_files"
    
    part = str(patient_dir).split("/")[-1]
    output_images_dir = Path(f"/home/imran-nasyrov/ainur_sct_json/{part}/images")
    output_annotations_dir = Path(f"/home/imran-nasyrov/ainur_sct_json/{part}/annotations")
    print("output_images_dir", output_images_dir)
    print("output_annotations_dir", output_annotations_dir)

    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)
            
    if not os.path.exists(output_annotations_dir):
        os.makedirs(output_annotations_dir)

    annotations = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "hemorrhagic stroke", "supercategory": "object"},
            {"id": 2, "name": "hemorrhagic stroke", "supercategory": "object"},
            {"id": 3, "name": "hemorrhagic stroke", "supercategory": "object"},
            {"id": 4, "name": "hemorrhagic stroke", "supercategory": "object"}
        ]
    }

    # Load masks from NPZ file once
    npz_file = Path(f"{masks_dir}/masks.npz")
    if not npz_file.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_file}")
    
    mask_data = np.load(npz_file)
    masks = mask_data.get('arr_0')
    
    if masks is None:
        raise ValueError(f"No masks found in NPZ file: {npz_file}")

    num_slices, num_classes, height, width = masks.shape
    print(f"Original Mask dimensions: {masks.shape}")

    # Desired new dimensions
    new_height, new_width = 512, 512

    # Resize all masks to the new dimensions
    resized_masks = np.zeros((num_slices, num_classes, new_height, new_width), dtype=masks.dtype)
    for slice_index in range(num_slices):
        for class_index in range(num_classes):
            resized_masks[slice_index, class_index] = cv2.resize(masks[slice_index, class_index], (new_width, new_height))

    print("resized_masks shape", resized_masks.shape)
    
    annotation_id = 1
    for image_id, dcm_file in enumerate(sorted(Path(images_dir).iterdir()), start=1):
        if dcm_file.suffix.lower() != '.dcm':
            continue

        # Read the DICOM file
        dicom_data = pydicom.dcmread(dcm_file)
        
        # Define the file_name as the DICOM file name
        file_name = dcm_file.name

        # Collect image information
        image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": new_width,
            "height": new_height,
        }
        annotations["images"].append(image_info)

        # Copy DICOM file to the new directory
        output_image_path = output_images_dir / file_name
        output_image_path.write_bytes(dcm_file.read_bytes())

        # Ensure the index is within the range of the resized masks array
        if image_id - 1 < num_slices:
            for class_id in range(num_classes):
                mask = resized_masks[image_id - 1, class_id, :, :]

                # Find contours using OpenCV
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                segmentation = []
                for contour in contours:
                    if len(contour) > 0:
                        segmentation.append(contour.flatten().tolist())

                # Find bounding box
                y_indices, x_indices = np.where(mask > 0)
                if len(x_indices) == 0 or len(y_indices) == 0:
                    continue

                x_min = int(x_indices.min())
                x_max = int(x_indices.max())
                y_min = int(y_indices.min())
                y_max = int(y_indices.max())

                bbox_width = x_max - x_min + 1
                bbox_height = y_max - y_min + 1

                # Calculate the area
                area = bbox_width * bbox_height

                # Create annotation for this mask
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,  # category_id should be from 1 to 4
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "area": area,
                    "segmentation": segmentation,
                    "iscrowd": 0
                }
                annotations["annotations"].append(annotation)
                annotation_id += 1

    # Save annotations as JSON
    annotations_file_path = output_annotations_dir / "instances_default.json"
    with annotations_file_path.open("w") as f:
        json.dump(annotations, f)#, indent=4)

# Define the base directory where all patient data is stored
base_dir = Path("/mnt/netstorage/Medicine/Medical/ainur_sct")

# Iterate over each patient folder
for patient_folder in base_dir.iterdir():
    if patient_folder.is_dir():
        create_coco_annotations(patient_folder)
