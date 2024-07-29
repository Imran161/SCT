import os

# Указываем путь к папке
base_path = '/home/imran/json_pochki/sagital'

# Проходим по всем элементам в указанной папке
for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)
    
    # Проверяем, что это действительно папка
    if os.path.isdir(folder_path):
        # Создаем новое имя, добавляя 'frontal_' в начало
        new_folder_name = 'sagital_' + folder_name
        new_folder_path = os.path.join(base_path, new_folder_name)
        
        # Переименовываем папку
        # print("folder_path", folder_path)
        # print("new_folder_path", new_folder_path)
        
        os.rename(folder_path, new_folder_path)
        print(f'Папка {folder_name} переименована в {new_folder_name}')
