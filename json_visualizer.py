import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from json_handler import JsonHandler


class ContourVisualizer:
    colors = [
        ((251, 206, 177), "Абрикосовым"),
        ((127, 255, 212), "Аквамариновым"),
        ((255, 36, 0), "Алым"),
        ((153, 102, 204), "Аметистовым"),
        ((153, 0, 102), "Баклажановым"),
        ((48, 213, 200), "Бирюзовым"),
        ((152, 251, 152), "Бледно зеленым"),
        ((213, 113, 63), "Ванильным"),
        ((100, 149, 237), "Васильковым"),
        ((34, 139, 34), "Зелёный лесной"),
        ((0, 0, 255), "Синий"),
        ((75, 0, 130), "Индиго"),
        ((255, 0, 255), "Чёрный"),
        ((0, 51, 153), "Маджента"),
        ((65, 105, 225), "Королевский синий"),
        ((255, 255, 0), "Жёлтый"),
        ((255, 69, 0), "Оранжево-красный"),
        ((255, 0, 0), "Темно синим"),
        ((0, 51, 153), "Красный"),
        ((255, 215, 0), "Золотой"),
        ((250, 128, 114), "Лососевый"),
        ((255, 99, 71), "Томатный"),
        ((255, 215, 0), "Золотой"),
        ((0, 139, 139), "Тёмный циан"),
        ((0, 255, 255), "Морская волна"),
    ]

    def __init__(self, list_of_name_out_classes):
        self.list_of_name_out_classes = list_of_name_out_classes

    @staticmethod
    def get_all_files(directory):
        all_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)
        return all_files

    def save_to_papki(self, path, number_papki, second_chislo, sct_coco):
        for k in sct_coco.all_img_list:
            result = sct_coco[k]
            image = result["images"]
            label = result["labels"]
            label = label[1:]
            if label.max().item() != 0:
                clas = label.argmax().item() + 1
                mask = result["masks"]
                rgb_image = result["rgb_image"]
                mask = mask.detach().numpy()

                rgb_image_plt = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                plt.imshow(rgb_image_plt)

                for i in range(np.shape(mask)[0]):
                    contours, h = cv2.findContours(
                        mask[i].astype(int).astype(np.uint8),
                        cv2.RETR_TREE,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    rgb_image_plt = cv2.drawContours(
                        rgb_image_plt,
                        contours,
                        -1,
                        (
                            self.colors[i][0][0],
                            self.colors[i][0][1],
                            self.colors[i][0][2],
                        ),
                        2,
                    )
                    if np.max(mask[i]) == 1 and i != 0:
                        text = self.list_of_name_out_classes[i]
                        plt.text(10, 20, f"label class: {text}", color=(1, 0, 0))
                plt.imshow(rgb_image_plt)
                plt.axis("off")

                photo_path = (
                    f"/home/imran-nasyrov/sct_project/sct_data/output_images/{clas}"
                )
                if not os.path.exists(photo_path):
                    os.makedirs(photo_path)

                all_files = self.get_all_files(photo_path)
                if len(all_files) < 100:
                    plt.savefig(
                        f"{photo_path}/output_image_{number_papki}_{second_chislo}_{k}.jpg"
                    )
                    plt.clf()
                    plt.close()


if __name__ == "__main__":
    list_of_name_out_classes = [
        "0",
        "Внутримозговое кровозлияние",
        "Субарахноидальное кровозлияние",
        "Cубдуральное кровозлияние",
        "Эпидуральное кровозлияние",
    ]

    тут надо на словарь заменять
    sct_coco = Universal_json_Segmentation_Dataset(
        json_file_path=path + "/",
        delete_list=[],
        base_classes=SCT_base_classes,
        out_classes=SCT_base_classes,
        delete_null=False,  # Fasle всегда
        resize=(512, 512),
        dataloader=True,
        recalculate=False,  # оставить True
        train_val_probs=100,
    )

    visualizer = ContourVisualizer(list_of_name_out_classes)
    visualizer.save_to_papki(path, number_papki, second_chislo, sct_coco)


короче в классе рисовашка была такая, оттуда убрал, а потом понял что мне
вообще и этот файл не нужен, потому что у меня есть sct_val.py

   def show_me_contours(self, idx):
        gray_image, mask, rgb_image = self.__getitem__(idx, contures=True)
        plt.rcParams["figure.figsize"] = [12, 12]
        plt.rcParams["figure.autolayout"] = True
        k = 0
        for i in range(np.shape(mask)[0]):
            contours, h = cv2.findContours(
                mask[i].astype(int).astype(np.uint8),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            # print("contours", contours)
            rgb_image = cv2.drawContours(
                rgb_image,
                contours,
                -1,
                (self.colors[i][0][0], self.colors[i][0][1], self.colors[i][0][2]),
                2,
            )

            if np.max(mask[i]) == 1 and i != 0:
                text = self.list_of_name_out_classes[i] + " " + str(np.max(mask[i]))
                # для 255 снимка такой вывод
                # print("i", i) # i 3
                # print("list_of_name_out_classes", self.list_of_name_out_classes) # ['фон', '1', '2', '3', '4', '5']
                # print("list_of_name_out_classes[i]", self.list_of_name_out_classes[i]) # 3
                # print("text", text) # 1 выводит
                plt.text(
                    2000,
                    k,
                    text,
                    color=(
                        self.colors[i][0][0] / 255,
                        self.colors[i][0][1] / 255,
                        self.colors[i][0][2] / 255,
                    ),
                )
                # так попробую
                cv2.putText(
                    rgb_image,
                    f"label class: {text}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                k += 50

        # сохраню фотки, сделаю папки по классам
        # если патологии нет, то ошибка будет, потому что text не существует
        # print("text", text[0])
        if not os.path.exists(
            "/home/imran-nasyrov/sct_project/sct_data/output_images/" + text[0]
        ):
            os.mkdir(
                "/home/imran-nasyrov/sct_project/sct_data/output_images/" + text[0]
            )

        all_files = get_all_files(
            "/home/imran-nasyrov/sct_project/sct_data/output_images/" + text[0]
        )
        if len(all_files) <= 5:
            cv2.imwrite(
                f"/home/imran-nasyrov/sct_project/sct_data/output_images/{text[0]}/output_image_{idx}.jpg",
                rgb_image,
            )

    def show_me_contours(self, idx):
        gray_image, mask, rgb_image = self.__getitem__(idx, contures=True)
        plt.rcParams["figure.figsize"] = [12, 12]
        plt.rcParams["figure.autolayout"] = True
        k = 0
        for i in range(np.shape(mask)[0]):
            contours, h = cv2.findContours(
                mask[i].astype(int).astype(np.uint8),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            # print("contours", contours)
            rgb_image = cv2.drawContours(
                rgb_image,
                contours,
                -1,
                (self.colors[i][0][0], self.colors[i][0][1], self.colors[i][0][2]),
                2,
            )

            if np.max(mask[i]) == 1 and i != 0:
                text = self.list_of_name_out_classes[i] + " " + str(np.max(mask[i]))
                # для 255 снимка такой вывод
                # print("i", i) # i 3
                # print("list_of_name_out_classes", self.list_of_name_out_classes) # ['фон', '1', '2', '3', '4', '5']
                # print("list_of_name_out_classes[i]", self.list_of_name_out_classes[i]) # 3
                # print("text", text) # 1 выводит
                plt.text(
                    2000,
                    k,
                    text,
                    color=(
                        self.colors[i][0][0] / 255,
                        self.colors[i][0][1] / 255,
                        self.colors[i][0][2] / 255,
                    ),
                )
                # так попробую
                cv2.putText(
                    rgb_image,
                    f"label class: {text}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                k += 50

        # сохраню фотки, сделаю папки по классам
        # если патологии нет, то ошибка будет, потому что text не существует
        # print("text", text[0])
        if not os.path.exists(
            "/home/imran-nasyrov/sct_project/sct_data/output_images/" + text[0]
        ):
            os.mkdir(
                "/home/imran-nasyrov/sct_project/sct_data/output_images/" + text[0]
            )

        all_files = get_all_files(
            "/home/imran-nasyrov/sct_project/sct_data/output_images/" + text[0]
        )
        if len(all_files) <= 5:
            cv2.imwrite(
                f"/home/imran-nasyrov/sct_project/sct_data/output_images/{text[0]}/output_image_{idx}.jpg",
                rgb_image,
            )
