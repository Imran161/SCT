import os

print("int", int("055")) # 55

# Путь к корневой директории
root_dir = '/home/imran-nasyrov/sct_project/sct_data/FINAL_CONVERT_NEW'

# Получаем список всех поддиректорий в корневой директории
sub_dirs = next(os.walk(root_dir))[1]

# Инициализируем пустой список для хранения имен главных папок без подпапки annotations
main_folders = set()

# Проверяем наличие папки annotations в каждой поддиректории первого уровня
for sub_dir in sub_dirs:
    # print("sub_dir", sub_dir)
    sub_dir_path = os.path.join(root_dir, sub_dir)
    sub_sub_dirs = next(os.walk(sub_dir_path))[1]
    for sub_sub_dir in sub_sub_dirs:
        # print("sub_sub_dir", sub_sub_dir)
        sub_sub_dir_path = os.path.join(sub_dir_path, sub_sub_dir)
        # print("sub_sub_dir_path", sub_sub_dir_path)
        # print("os.listdir(sub_sub_dir_path)", os.listdir(sub_sub_dir_path))
        if 'annotations' not in os.listdir(sub_sub_dir_path):
            main_folders.add(sub_dir)

# main_folders содержит имена подпапок первого уровня без подпапки annotations
print(main_folders)
print("main_folders", len(main_folders))

# вот эти папки мне вывелись, но у них есть лэйблы в Labels но почему то их нет в FINAL_CONVERT_OLD 
# буду переделывать в FINAL_CONVERT
# 201000265
# 201206791



# # тут найду папки со словом желудок и потом удалю их
# import os
# import shutil

# # Путь к директории
# directory_path = "/mnt/datastore/Medical/stomach_paths"

# # Получаем список всех файлов и папок в директории
# contents = os.listdir(directory_path)

# # Фильтруем список, оставляя только папки, в названиях которых есть слово "кишечник"
# filtered_folders = [folder for folder in contents if os.path.isdir(os.path.join(directory_path, folder)) and "кишечник" in folder]

# print(filtered_folders)
# print("len filtered_folders", len(filtered_folders))

# # Удаляем каждую папку из отфильтрованного списка
# for folder in filtered_folders:
#     folder_path = os.path.join(directory_path, folder)
#     # print("folder_path", folder_path)
    
#     # это удалит папки!!! закоменчу на всякий случай
#     # shutil.rmtree(folder_path)

# print("Папки успешно удалены.")

