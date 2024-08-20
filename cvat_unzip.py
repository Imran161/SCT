import os
import zipfile

# Путь к исходной папке с архивами
source_dir = "/home/imran-nasyrov/cvat/"

# Путь к папке, куда будут распаковываться архивы
destination_dir = "/home/imran-nasyrov/cvat_unzip/"

# Проверяем, существует ли целевая директория, если нет — создаем её
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Проходимся по всем файлам в исходной папке
for filename in os.listdir(source_dir):
    if filename.endswith(".zip"):
        # Определяем полный путь к архиву
        file_path = os.path.join(source_dir, filename)

        # Определяем имя папки, которая будет создана для распаковки
        folder_name = filename[:-4]  # Убираем '.zip' из имени файла
        folder_path = os.path.join(destination_dir, folder_name)

        # Создаем папку для распакованных файлов
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Распаковываем архив в созданную папку
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(folder_path)

        print(f"Архив {filename} успешно распакован в {folder_path}")

print("Все архивы успешно распакованы.")
