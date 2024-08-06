import os
import json
import pydicom
import numpy as np
import cv2
from pathlib import Path
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut


def visualize_annotations(patient_dir, output_dir):
    images_dir = Path(f"{patient_dir}/images")
    annotations_path = Path(f"{patient_dir}/annotations/instances_default.json")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # Create a map of image id to image info
    images_info = {image['id']: image for image in annotations['images']}
    
    # Create a map of image id to annotations
    annotations_map = {}
    for annotation in annotations['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_map:
            annotations_map[image_id] = []
        annotations_map[image_id].append(annotation)
    
    for image_info in images_info.values():
        image_id = image_info['id']
        file_name = image_info['file_name']
        image_path = images_dir / file_name

        # Load the DICOM image
        dicom_data = pydicom.dcmread(image_path)
        # image = dicom_data.pixel_array

        img = apply_modality_lut(apply_voi_lut(dicom_data.pixel_array, dicom_data), dicom_data)
        image = img
        # Convert to 8-bit image if necessary
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Convert to RGB for visualization
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Check if there are annotations for the current image
        if image_id not in annotations_map:
            continue

        # List to hold all contours for the current image
        all_contours = []

        # Draw contours for each annotation
        has_masks = False
        for annotation in annotations_map.get(image_id, []):
            for segmentation in annotation['segmentation']:
                contour = np.array(segmentation).reshape((-1, 1, 2)).astype(np.int32)
                all_contours.append(contour)
                has_masks = True

        # Draw all contours in red color
        if has_masks:
            cv2.drawContours(image_rgb, all_contours, -1, (255, 0, 0), 2)
            
            # Save the image with contours only if there are masks
            output_image_path = output_dir / f"{file_name}.png"
            cv2.imwrite(str(output_image_path), image_rgb)



def draw_masks_on_images(json_dir, output_dir):
    # Read the annotations JSON file
    json_path = Path(json_dir) / "annotations/instances_default.json"
    with open(json_path, "r") as f:
        annotations_data = json.load(f)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each image
    for image_info in annotations_data["images"]:
        image_id = image_info["id"]
        file_name = image_info["file_name"]
        width = image_info["width"]
        height = image_info["height"]

        # Load the corresponding DICOM file
        dicom_path = Path(json_dir) / "images" / file_name
        dicom_data = pydicom.dcmread(dicom_path)
        
        # Apply VOI LUT and Modality LUT to the image
        img = apply_modality_lut(apply_voi_lut(dicom_data.pixel_array, dicom_data), dicom_data)
        
        # Normalize the image
        normalized_img = 255.0 * (img - np.min(img)) / (np.max(img) - np.min(img))
        color_img = cv2.merge([normalized_img] * 3)  # Convert to color image (3 channels)

        # Draw annotations
        for annotation in annotations_data["annotations"]:
            if annotation["image_id"] == image_id:
                segmentation = annotation["segmentation"]
                if segmentation:
                    # Convert the segmentation points to a numpy array and reshape to (-1, 2)
                    points = np.array(segmentation[0]).reshape(-1, 2).astype(np.int32)
                    # Draw the contours in red (BGR format: (0, 0, 255))
                    cv2.drawContours(color_img, [points], -1, (0, 0, 255), 2)

        # Save the resulting image if there is at least one mask
        if np.any(color_img):
            output_image_path = Path(output_dir) / f"{Path(file_name).stem}_with_mask.png"
            cv2.imwrite(str(output_image_path), color_img)


# Specify the input and output directories
patient_dir = Path("/home/imran-nasyrov/ainur_sct_json/") #1.2.643.5.1.13.13.12.2.77.8252.14110807150508020609090907121401")
output_dir = "/home/imran-nasyrov/output_guts2"


# Run the visualization
for patient_folder in patient_dir.iterdir():
    print("patient_folder", patient_folder)
    part = str(patient_folder).split("/")[-1]
    # if patient_folder.is_dir():
    visualize_annotations(patient_folder, f"{output_dir}/{part}")
