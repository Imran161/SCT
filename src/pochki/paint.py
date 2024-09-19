import os
import numpy as np
from pycocotools.coco import COCO
import cv2
from pycocotools import mask as maskUtils




# def draw_contour(image, mask, color):
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(image, contours, -1, color, 2)  # Draw all contours with specified color and thickness
#     return image, contours

# def apply_mask_contours(image, mask, color, class_name):
#     image_with_contours, contours = draw_contour(image, mask, color)
    
#     # Draw class name in the top-right corner of the image
#     cv2.putText(image_with_contours, class_name, (image.shape[1] - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

#     return image_with_contours

# # def apply_mask_contours(image, mask, color, class_name):
# #     image_with_contours, contours = draw_contour(image, mask, color)
    
# #     # Find bounding box for each contour
# #     for contour in contours:
# #         x, y, w, h = cv2.boundingRect(contour)
        
# #         # Calculate text size
# #         (text_width, text_height), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
# #         # Draw class name at top-right corner of bounding box
# #         cv2.putText(image_with_contours, class_name, (x + w - text_width - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

# #     return image_with_contours


# # Path to the directory containing JSON annotations
# json_dir = "/home/alexskv/data/json_pochki/"

# # Output directory for saving images with contours
# output_root = "/home/alexskv/pochki/nrrd_output_images_cv2"

# # Create output directory if it doesn't exist
# os.makedirs(output_root, exist_ok=True)

# class_colors = {
#     0: (0, 255, 0),
#     1: (0, 0, 255),
#     2: (255, 0, 0),
#     3: (0, 255, 255),
#     4: (255, 0, 255),
#     5: (255, 255, 0),
#     6: (128, 0, 128),
#     7: (128, 128, 0),
#     8: (0, 128, 128),
#     9: (128, 128, 128),
#     10: (0, 0, 128),
#     11: (0, 128, 0)
# }

# # Iterate over all folders in json_dir
# for folder in os.listdir(json_dir):
#     folder_path = os.path.join(json_dir, folder)
#     if not os.path.isdir(folder_path):
#         continue
    
#     # Path to the COCO annotations file
#     annFile = os.path.join(folder_path, "annotations/instances_default.json")

#     # Load COCO annotations
#     coco = COCO(annFile)
#     cats = coco.loadCats(coco.getCatIds())
#     cat_names = [cat['name'] for cat in cats]

#     # Get all image IDs
#     imgIds = coco.getImgIds()

#     # Output folder for saving images with contours
#     output_folder = os.path.join(output_root, f"output_{folder}")
#     os.makedirs(output_folder, exist_ok=True)

#     for img_id in imgIds:
#         img = coco.loadImgs(img_id)[0]
#         img_path = os.path.join(folder_path, "images", img['file_name'])
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=False)
#         # print("annIds", annIds)
#         anns = coco.loadAnns(annIds)

#         if not anns:
#             continue  # Skip images without masks

#         for ann in anns:
#             mask = coco.annToMask(ann)
#             class_id = ann['category_id']

#             if class_id in class_colors:
#                 color = class_colors[class_id]
#             else:
#                 color = (0, 0, 0)  # Default to black if color not defined
            
#             class_name = coco.loadCats(class_id)[0]['name']
#             image = apply_mask_contours(image, mask, color, class_name)
            
#             # print("img_id", img_id)
#             # values, counts = np.unique(image, return_counts=True)
#             # print("Уникальные значения и их количество в image")
#             # for value, count in zip(values, counts):
#             #     print(f"Value: {value}, Count: {count}")
    

#         # Formulate the save path
#         mask_filename = os.path.splitext(img['file_name'])[0] + '_contours_with_labels.png'
#         save_path = os.path.join(output_folder, mask_filename)

#         # Save the image with contours and class labels
#         cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
#         print(f"Saved: {save_path}")

# print("Saving process completed.")





def draw_contour(image, mask, color):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, 2)  # Draw all contours with specified color and thickness
    return image, contours

def apply_mask_contours(image, mask, color, class_name, class_index):
    image_with_contours, contours = draw_contour(image, mask, color)
    
    # Draw class name below the contours
    cv2.putText(image_with_contours, class_name, (10, 30 + class_index * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)

    return image_with_contours

# Path to the directory containing JSON annotations
json_dir = "/home/alexskv/data/json_pochki/"

# Output directory for saving images with contours
output_root = "/home/alexskv/pochki/nrrd_output_images_cv2"

# Create output directory if it doesn't exist
os.makedirs(output_root, exist_ok=True)

class_colors = {
    0: (0, 255, 0),
    1: (0, 0, 255),
    2: (255, 0, 0),
    3: (0, 255, 255),
    4: (255, 0, 255),
    5: (255, 255, 0),
    6: (128, 0, 128),
    7: (128, 128, 0),
    8: (0, 128, 128),
    9: (128, 128, 128),
    10: (0, 0, 128),
    11: (0, 128, 0)
}


kidney = ["right_kidney_ID1", "left_kidney_ID5"]

kidney_segments = [
    "right_kidney_upper_segment_ID2",
    "right_kidney_middle_segment_ID3",
    "right_kidney_lower_segment_ID4",
    "left_kidney_upper_segment_ID6",
    "left_kidney_middle_segment_ID7",
    "left_kidney_lower_segment_ID8",
]

malignant_tumors = ["malignant_tumor_ID9", "benign_tumor_ID10", "cyst_ID11", "abscess_ID12"]


# Iterate over all folders in json_dir
for folder in os.listdir(json_dir):
    folder_path = os.path.join(json_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    
    # if "Zaytseva TI_2 Non Contrast  3.0  I41s  4" in folder_path:
    # Path to the COCO annotations file
    annFile = os.path.join(folder_path, "annotations/instances_default.json")

    # Load COCO annotations
    coco = COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())
    cat_names = [cat['name'] for cat in cats]

    # Get all image IDs
    imgIds = coco.getImgIds()

    # Output folder for saving images with contours
    output_folder = os.path.join(output_root, f"output_{folder}")
    os.makedirs(output_folder, exist_ok=True)

    for img_id in imgIds:
        img = coco.loadImgs(img_id)[0]
        img_path = os.path.join(folder_path, "images", img['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_kidney = image.copy()
        image_segments = image.copy()
        image_tumors = image.copy()

        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=False)
        # print("annIds", annIds)
        anns = coco.loadAnns(annIds)

        if not anns:
            continue  # Skip images without masks

        for idx, ann in enumerate(anns):
            # if coco.loadCats(ann['category_id'])[0]['name'] in ["malignant_tumor_ID9"]: #еще один сделать с сегментами и с остальными и сделать три картинки в ряд
            class_name = coco.loadCats(ann['category_id'])[0]['name']
            mask = coco.annToMask(ann)
            class_id = ann['category_id']

            #     if class_id in class_colors:
            #         color = class_colors[class_id]
            #     else:
            #         color = (0, 0, 0)  # Default to black if color not defined
                
            #     class_name = coco.loadCats(class_id)[0]['name']
            #     print("class_name = coco.loadCats(class_id)[0]", coco.loadCats(class_id)[0])
            #     print("class_name", class_name)
            #     print("id", ann["id"])
            #     image = apply_mask_contours(image, mask, color, class_name, idx)

            if class_name in kidney:
                color = class_colors.get(class_id, (0, 0, 0))
                image_kidney = apply_mask_contours(image_kidney, mask, color, class_name, idx)
            elif class_name in kidney_segments:
                color = class_colors.get(class_id, (0, 0, 0))
                image_segments = apply_mask_contours(image_segments, mask, color, class_name, idx)
            elif class_name in malignant_tumors:
                color = class_colors.get(class_id, (0, 0, 0))
                image_tumors = apply_mask_contours(image_tumors, mask, color, class_name, idx)
        
        concatenated_image = np.concatenate((image_kidney, image_segments, image_tumors), axis=1)

        

        # # Formulate the save path
        # mask_filename = os.path.splitext(img['file_name'])[0] + '_contours_with_labels.png'
        # save_path = os.path.join(output_folder, mask_filename)

        # # Save the image with contours and class labels
        # cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # print(f"Saved: {save_path}")
        
        mask_filename = os.path.splitext(img['file_name'])[0] + '_contours_with_labels.png'
        save_path = os.path.join(output_folder, mask_filename)

        # Save the concatenated image
        cv2.imwrite(save_path, cv2.cvtColor(concatenated_image, cv2.COLOR_RGB2BGR))
        print(f"Saved: {save_path}")

print("Saving process completed.")

