import os
import json
from tqdm import tqdm

# Путь к директории
directory_path = "/mnt/datastore/Medical/stomach_json"
# directory_path = "/home/imran-nasyrov/sct_project/sct_data/test_r_json"


# Функция для замены поля "categories"
def replace_categories(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
        data["categories"] = [
            {"id": 24, "name": "0", "supercategory": ""},
            {"id": 0, "name": "SE", "supercategory": ""},
            {"id": 1, "name": "GT", "supercategory": ""},
            {"id": 2, "name": "NG", "supercategory": ""},
            {"id": 3, "name": "F", "supercategory": ""},
            {"id": 4, "name": "IM", "supercategory": ""},
            {"id": 5, "name": "LT", "supercategory": ""},
            {"id": 6, "name": "GINL", "supercategory": ""},
            {"id": 7, "name": "GINH", "supercategory": ""},
            {"id": 8, "name": "TACG1", "supercategory": ""},
            {"id": 9, "name": "TACG2", "supercategory": ""},
            {"id": 10, "name": "TACG3", "supercategory": ""},
            {"id": 11, "name": "PACG1", "supercategory": ""},
            {"id": 12, "name": "PACG2", "supercategory": ""},
            {"id": 13, "name": "MPAC", "supercategory": ""},
            {"id": 14, "name": "PCC", "supercategory": ""},
            {"id": 15, "name": "PCC-NOS", "supercategory": ""},
            {"id": 16, "name": "MAC1", "supercategory": ""},
            {"id": 17, "name": "MAC2", "supercategory": ""},
            {"id": 18, "name": "ACLS", "supercategory": ""},
            {"id": 19, "name": "HAC", "supercategory": ""},
            {"id": 20, "name": "ACFG", "supercategory": ""},
            {"id": 21, "name": "SCC", "supercategory": ""},
            {"id": 22, "name": "NDC", "supercategory": ""},
            {"id": 23, "name": "NED", "supercategory": ""},
        ]

    with open(file_path, "w") as file:
        json.dump(data, file)


# Обход всех подпапок
for root, dirs, files in os.walk(directory_path):
    with tqdm(total=len(dirs), desc="Making files") as pbar_dirs:
        for dir in dirs:
            # Путь к файлу instances_default.json
            file_path = os.path.join(root, dir, "annotations", "instances_default.json")

            if os.path.exists(file_path):
                replace_categories(file_path)
            pbar_dirs.update(1)
