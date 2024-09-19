import os
import json
import cv2
import numpy as np
from pathlib import Path

def draw_contours_on_images(patient_name, images_dir, annotations_file, output_base_dir):
    # Load the COCO annotations
    with open(annotations_file, "r") as f:
        annotations = json.load(f)
    
    # Create output directory for the patient
    output_patient_dir = Path(output_base_dir) / patient_name
    output_patient_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to map image_id to image metadata
    image_id_to_filename = {img["id"]: img["file_name"] for img in annotations["images"]}

    # Iterate over annotations and draw contours
    for annotation in annotations["annotations"]:
        image_id = annotation["image_id"]
        file_name = image_id_to_filename[image_id]
        image_path = Path(images_dir) / file_name
        
        # Load the grayscale image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error loading image: {image_path}")
            continue

        # Convert the image to BGR to draw colored contours
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Draw each contour
        for segmentation in annotation["segmentation"]:
            contour = np.array(segmentation).reshape((-1, 2)).astype(np.int32)
            cv2.polylines(image_bgr, [contour], isClosed=True, color=(0, 0, 255), thickness=2)  # Red color for contours

        # Save the output image
        output_image_path = output_patient_dir / file_name
        cv2.imwrite(str(output_image_path), image_bgr)
    
    print(f"Processed {patient_name}, output saved to {output_patient_dir}")

def process_all_patients_json(base_json_dir, output_base_dir):
    for patient_folder in Path(base_json_dir).iterdir():
        if patient_folder.is_dir():
            images_dir = patient_folder / "images"
            annotations_file = patient_folder / "annotations" / "instances_default.json"

            if not annotations_file.exists():
                print(f"Annotations file not found for {patient_folder}, skipping...")
                continue

            try:
                draw_contours_on_images(patient_folder.name, images_dir, annotations_file, output_base_dir)
            except Exception as e:
                print(f"Error processing {patient_folder}: {e}")

# Base directories
base_json_dir = Path("/home/imran-nasyrov/mrt_kaggle_json")
output_base_dir = Path("/home/imran-nasyrov/output_guts2")

# Process all patients
process_all_patients_json(base_json_dir, output_base_dir)
