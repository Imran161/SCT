# import os
# import json
# import cv2

# # Указать пути к папкам
# src_folder = 'test_sinusite_jsonl'
# dst_folder = 'jsonl_out'
# # src_folder = '/home/imran-nasyrov/cvat_phrase'
# # dst_folder = 'florence_dataloader'


# # Создать целевую папку, если она не существует
# os.makedirs(dst_folder, exist_ok=True)

# # Пройтись по всем подкаталогам в папке sinusite_jsonl
# for task_folder in os.listdir(src_folder):
#     # if task_folder in "task_sinusite_data_29_11_23_1_st_sin_labeling":
#     #     print("task_folder", task_folder)
#     task_path = os.path.join(src_folder, task_folder)
#     if not os.path.isdir(task_path):
#         continue

#     images_path = os.path.join(task_path, 'images')
#     jsonl_file_path = os.path.join(task_path, 'annotations.jsonl')

#     # Создать соответствующий каталог в целевой папке
#     dst_task_path = os.path.join(dst_folder, task_folder)
#     os.makedirs(dst_task_path, exist_ok=True)

#     # Прочитать JSONL файл
#     with open(jsonl_file_path, 'r') as f:
#         lines = f.readlines()

#     for line in lines:
#         data = json.loads(line)
#         image_filename = data['image']
#         suffix = data['suffix']
#         prefix = data['prefix']
        
#         # Загрузить изображение
#         image_path = os.path.join(images_path, image_filename)
#         image = cv2.imread(image_path)
#         if image is None:
#             print("image_path", image_path)
#             continue
        
#         # Получить размеры изображения
#         height, width, _ = image.shape

#         # Разбить suffix на части
#         parts = suffix.split('<loc_')
#         class_name = parts[0].rstrip('<>')

#         for i in range(1, len(parts), 4):
#             if i + 3 < len(parts):
#                 try:
#                     # Преобразовать координаты обратно к оригинальным размерам
#                     x1 = int(parts[i].split('>')[0]) * width // 1000
#                     y1 = int(parts[i + 1].split('>')[0]) * height // 1000
#                     x2 = int(parts[i + 2].split('>')[0]) * width // 1000
#                     y2 = int(parts[i + 3].split('>')[0]) * height // 1000

#                     # Нарисовать прямоугольник
#                     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

#                     # Подписать класс
#                     cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                 except (IndexError, ValueError) as e:
#                     print(f"Error processing bounding box: {e}")

#         # Сохранить изображение в целевой папке
#         dst_image_path = os.path.join(dst_task_path, image_filename)

#         os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
#         cv2.imwrite(dst_image_path, image)




# вот тут будет езе подпись префикса и суффикса 

import os
import json
import cv2

# Указать пути к папкам
# src_folder = '/home/imran-nasyrov/cvat_phrase'
# dst_folder = 'florence_dataloader'

src_folder = '/home/imran-nasyrov/cvat_mini'
dst_folder = 'jsonl_out'

# Создать целевую папку, если она не существует
os.makedirs(dst_folder, exist_ok=True)

# Пройтись по всем подкаталогам в папке sinusite_jsonl
for task_folder in os.listdir(src_folder):
    task_path = os.path.join(src_folder, task_folder)
    if not os.path.isdir(task_path):
        continue

    images_path = os.path.join(task_path, 'images')
    jsonl_file_path = os.path.join(task_path, 'annotations.jsonl')

    # Создать соответствующий каталог в целевой папке
    dst_task_path = os.path.join(dst_folder, task_folder)
    os.makedirs(dst_task_path, exist_ok=True)

    # Прочитать JSONL файл
    with open(jsonl_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        data = json.loads(line)
        image_filename = data['image']
        suffix = data['suffix']
        prefix = data['prefix']
        
        # Загрузить изображение
        image_path = os.path.join(images_path, image_filename)
        # print("image_filename", image_filename)
        # print("image_path", image_path)
        # break
        image = cv2.imread(image_path)
        if image is None:
            print("image_path", image_path)
            continue
        
        # Получить размеры изображения
        height, width, _ = image.shape

        # Разбить suffix на части
        parts = suffix.split('<loc_')
        class_name = parts[0].rstrip('<>')
        
        # Разделить строку по первому вхождению '<'
        suf = suffix.split('<', 1)

        # Добавить обратно '<' и вернуть часть строки, начиная с '<'
        result = '<' + suf[1] if len(suf) > 1 else ''
        
        # print("suffix", suffix)
        # print("result", result)
        # print("class_name", class_name)

        # Подготовить текст для отображения
        text_info = f"prefix: {prefix}\nclass: {class_name}\n"
        for i in range(1, len(parts), 4):
            if i + 3 < len(parts):
                try:
                    # Преобразовать координаты обратно к оригинальным размерам
                    x1 = int(parts[i].split('>')[0]) * width // 1000
                    y1 = int(parts[i + 1].split('>')[0]) * height // 1000
                    x2 = int(parts[i + 2].split('>')[0]) * width // 1000
                    y2 = int(parts[i + 3].split('>')[0]) * height // 1000

                    # Нарисовать прямоугольник
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Подписать класс
                    cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Добавить текст с координатами
                    # text_info += f"({x1}, {y1}) - ({x2}, {y2})\n"
                    text_info += f"coord: {result}\n"
                    
                except (IndexError, ValueError) as e:
                    print(f"Error processing bounding box: {e}")

        # Поместить текст в верхний левый угол
        y0, dy = 30, 20
        for i, line in enumerate(text_info.split('\n')):
            y = y0 + i * dy
            cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # # Сохранить изображение в целевой папке
        # dst_image_path = os.path.join(dst_task_path, image_filename)

        # os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
        # cv2.imwrite(dst_image_path, image)

         # Сформировать новое имя файла с добавлением класса
        base_name, ext = os.path.splitext(image_filename)
        new_image_filename = f"{base_name}_{class_name}{ext}"

        # Сохранить изображение в целевой папке
        dst_image_path = os.path.join(dst_task_path, new_image_filename)
        # print("dst_image_path", dst_image_path)
        os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
        cv2.imwrite(dst_image_path, image)