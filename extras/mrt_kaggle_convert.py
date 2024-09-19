import os
import json
import cv2
import numpy as np
from pathlib import Path

def create_coco_annotations_for_patient(patient_name, images_dir, masks_dir, output_base_dir):
    # Create output directories
    output_patient_dir = Path(output_base_dir) / patient_name
    output_images_dir = output_patient_dir / "images"
    output_annotations_dir = output_patient_dir / "annotations"
    
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize COCO structure
    annotations = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "vertebral_outline", "supercategory": "object"}]
    }
    
    annotation_id = 1
    
    # Iterate over images and corresponding masks
    image_files = sorted(os.listdir(images_dir))
    
    for image_id, image_file in enumerate(image_files):
        if not image_file.endswith(".png"):
            continue
        
        image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, image_file.replace('Slice', 'Labels_Slice'))  # Corresponding mask file

        if not os.path.exists(mask_path):
            print(f"Mask not found for {image_file}, skipping...")
            continue

        # Read image and mask
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        height, width = image.shape
        print(f"Processing {image_file}: {width}x{height}")

        # Save the image to the output directory
        output_image_path = output_images_dir / image_file
        cv2.imwrite(str(output_image_path), image)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Prepare annotations
        segmentation = []
        for contour in contours:
            if len(contour) > 0:
                segmentation.append(contour.flatten().tolist())

        if segmentation:
            x, y, w, h = cv2.boundingRect(mask)
            area = int(np.sum(mask > 0))
            
            annotation = {
                "id": annotation_id,
                "image_id": image_id + 1,  # COCO format expects 1-based IDs
                "category_id": 1,  # "vertebral_outline"
                "segmentation": segmentation,
                "area": area,
                "bbox": [x, y, w, h],
                "iscrowd": 0
            }
            annotations["annotations"].append(annotation)
            annotation_id += 1
        
        # Add image info to the annotations
        annotations["images"].append({
            "id": image_id + 1,
            "file_name": image_file,
            "width": width,
            "height": height
        })

    # Save annotations as JSON
    annotations_file_path = output_annotations_dir / "instances_default.json"
    with open(annotations_file_path, "w") as f:
        json.dump(annotations, f)

    print(f"Annotations saved for {patient_name}.")


def process_all_patients(data_dir, output_base_dir):
    patients = sorted(os.listdir(data_dir / "images"))
    
    for patient in patients:
        images_dir = Path(data_dir) / "images" / patient
        masks_dir = Path(data_dir) / "masks" / patient
        
        if not images_dir.is_dir() or not masks_dir.is_dir():
            continue

        try:
            create_coco_annotations_for_patient(patient, images_dir, masks_dir, output_base_dir)
        except Exception as e:
            print(f"Error processing {patient}: {e}")


# Directories for input and output data
data_dir = Path("/home/imran-nasyrov/мрт_позвонки_kaggle")
output_base_dir = Path("/home/imran-nasyrov/mrt_kaggle_json")

# Process all patients
process_all_patients(data_dir, output_base_dir)
