import os

import cv2
import numpy as np


class ImageVisualizer:
    def __init__(self, output_path):
        self.output_path = output_path
        self.image_counter = 0

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def const_thresh(self, pred_masks, threshold=0.5):
        rounded_masks = pred_masks >= threshold
        return np.ceil(rounded_masks)

    def auto_thresh(self, pred_masks):
        thresholded_masks = []
        for i in range(pred_masks.shape[0]):
            pred_mask = pred_masks[i]

            # Нормализация значений в диапазон от 0 до 255
            normalized_mask = (pred_mask * 255.0).astype(np.uint8)

            # Применение метода Оцу
            _, thresholded_mask = cv2.threshold(
                normalized_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Нормализация обратно в диапазон от 0 до 1
            thresholded_mask = thresholded_mask / 255.0

            # было
            # Нормализация значений в диапазон от 0 до 255
            # normalized_mask = cv2.normalize(pred_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            # # Применение метода Оцу
            # _, thresholded_mask = cv2.threshold(normalized_mask, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            thresholded_masks.append(thresholded_mask)

        return np.array(thresholded_masks)


    def visualize(self, images, true_masks, pred_masks, class_names_dict, colors, epoch=None, num_images_to_draw=None):
        true_masks = true_masks.detach().cpu().numpy()
        pred_masks = pred_masks.detach().cpu().numpy()

        if num_images_to_draw is None:
            num_images_to_draw = len(images)
        else:
            num_images_to_draw = min(num_images_to_draw, len(images))

        drawn_images = 0

        for i in range(len(images)):
            if drawn_images >= num_images_to_draw:
                break

            # Проверка, содержит ли настоящая маска ненулевые значения
            if np.sum(true_masks[i]) == 0:
                continue

            image = images[i][0].cpu().numpy()
            image = (image * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Определение настоящего класса и его цвета
            true_class_idx = np.argmax(np.amax(true_masks[i], axis=(1, 2)))
            true_class_name = class_names_dict.get(true_class_idx + 1, f"Class {true_class_idx + 1}")
            true_color = colors[true_class_idx][0]

            # Определение предсказанного класса и его цвета
            pred_class_idx = np.argmax(np.amax(pred_masks[i], axis=(1, 2)))
            pred_class_name = class_names_dict.get(pred_class_idx + 1, f"Class {pred_class_idx + 1}")
            pred_color = colors[pred_class_idx][0]

            # Извлечение настоящей и предсказанной масок
            true_mask = true_masks[i][true_class_idx]
            pred_mask = pred_masks[i][pred_class_idx]

            # Применение пороговой фильтрации к предсказанной маске
            thresholded_pred_mask = self.auto_thresh(pred_mask)

            # Создание изображений для настоящей и предсказанной масок
            true_contours_image = image.copy()
            pred_contours_image = image.copy()

            cv2.putText(true_contours_image, f"true_{true_class_name}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, true_color, 2, cv2.LINE_AA)
            cv2.putText(pred_contours_image, f"pred_{pred_class_name}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, pred_color, 2, cv2.LINE_AA)

            # Нахождение контуров для настоящей маски
            true_contours, _ = cv2.findContours((true_mask * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in true_contours:
                cv2.drawContours(true_contours_image, [contour], -1, true_color, 1)

            # Нахождение контуров для предсказанной маски
            pred_contours, _ = cv2.findContours((thresholded_pred_mask * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in pred_contours:
                cv2.drawContours(pred_contours_image, [contour], -1, pred_color, 1)

            # Объединение изображений в одно
            combined_image = np.hstack((true_contours_image, pred_contours_image))

            # Сохранение изображения
            if epoch is None:
                cv2.imwrite(f"{self.output_path}/image_{self.image_counter}.jpg", combined_image)
            else:
                cv2.imwrite(f"{self.output_path}/epoch_{epoch}_image_{self.image_counter}.jpg", combined_image)

            self.image_counter += 1
            drawn_images += 1
    
    
    def visualize_old(
        self, images, true_masks, pred_masks, class_names_dict, colors, epoch=None
    ):
        # print("type true_masks", type(true_masks))
        true_masks = true_masks.detach().cpu().numpy()
        # print("pred_masks type", type(pred_masks)) # <class 'torch.Tensor'>

        pred_masks = pred_masks.detach().cpu().numpy()
        pred_mask_no_tr = pred_masks.copy()
        pred_masks = self.auto_thresh(pred_masks)

        for i in range(len(images)):
            image = images[i][0].cpu().numpy()
            image = (image * 255).astype(np.uint8)

            true_image_with_contours = np.zeros(
                (image.shape[0], image.shape[1], 3), dtype=np.uint8
            )
            # Копируем одноканальное изображение в канал синего цвета (Blue)
            true_image_with_contours[:, :, 0] = image
            # Копируем одноканальное изображение в канал зеленого цвета (Green)
            true_image_with_contours[:, :, 1] = image
            # Копируем одноканальное изображение в канал красного цвета (Red)
            true_image_with_contours[:, :, 2] = image

            for j in range(np.shape(true_masks[i])[0]):
                # print("np.shape(true_masks[i])[0]", np.shape(true_masks[i])[0]) # 4
                color = (colors[j][0][0], colors[j][0][1], colors[j][0][2])
                contours, _ = cv2.findContours(
                    true_masks[i][j].astype(int).astype(np.uint8),
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                true_image_with_contours = cv2.drawContours(
                    true_image_with_contours, contours, -1, color, 2
                )

                # values, counts = np.unique(true_masks[i][j], return_counts=True)
                # for v, c in zip(values, counts):
                #     print(f"v:{v}, c:{c}")

                if np.max(true_masks[i][j]) == 1:
                    #     # text = list_of_name_out_classes[j]
                    # print("i, j",i, j)
                    text = class_names_dict[j + 1]
                    # print("text", text)
                    # plt.text(2000, k, text , color = (self.colors[i][0][0]/255, self.colors[i][0][1]/255, self.colors[i][0][2]/255))
                    cv2.putText(
                        true_image_with_contours,
                        f"true class: {text}",
                        (10, 20),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                # k+=50
            #     class_name = class_names_dict[j+1]
            # cv2.putText(true_image_with_contours, f"True Mask: {class_name}", (10, 20 + j * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            pred_image_with_contours = np.zeros(
                (image.shape[0], image.shape[1], 3), dtype=np.uint8
            )
            # Копируем одноканальное изображение в канал синего цвета (Blue)
            pred_image_with_contours[:, :, 0] = image
            # Копируем одноканальное изображение в канал зеленого цвета (Green)
            pred_image_with_contours[:, :, 1] = image
            # Копируем одноканальное изображение в канал красного цвета (Red)
            pred_image_with_contours[:, :, 2] = image

            for j in range(np.shape(pred_masks[i])[0]):
                color = (colors[j][0][0], colors[j][0][1], colors[j][0][2])
                contours, _ = cv2.findContours(
                    pred_masks[i][j].astype(int).astype(np.uint8),
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                # print("contours", contours)
                pred_image_with_contours = cv2.drawContours(
                    pred_image_with_contours, contours, -1, color, 2
                )

                # for class_idx, class_name in class_names_dict.items():
                #     cv2.putText(pred_image_with_contours, f"true class: {class_name}", (10, 20 * class_idx), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                if np.max(pred_masks[i][j]) == 1:
                    # text = list_of_name_out_classes[j]
                    # print("i, j",i, j)
                    text = class_names_dict[j + 1]
                    # print("text", text)
                    # plt.text(2000, k, text , color = (self.colors[i][0][0]/255, self.colors[i][0][1]/255, self.colors[i][0][2]/255))
                    cv2.putText(
                        pred_image_with_contours,
                        f"pred class: {text}",
                        (10, 20),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    # k+=50
            #     class_name = class_names_dict[j+1]  # Имя класса
            # cv2.putText(pred_image_with_contours, f"Pred Mask: {class_name}", (10, 20 + j * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            combined_image = np.concatenate(
                (true_image_with_contours, pred_image_with_contours), axis=1
            )

            values, counts = np.unique(true_masks[i], return_counts=True)
            if len(values) > 1:
                if epoch is None:
                    cv2.imwrite(f"{self.output_path}/image_{i}.jpg", combined_image)
                else:
                    cv2.imwrite(
                        f"{self.output_path}/epoch_{epoch}_image_{i}.jpg",
                        combined_image,
                    )

    # сохранить маску без трешхолда не на черном фоне
    # то есть без парога я рисую все маски вероятностей
    # потом я рисую контуры по оптимальному парогу либо 0.1, нахожу все контуры которые есть и каждый контур
    # в бинарные и нахожу контур и потом для каждого контура нахожу среднюю вероятность
    # то есть для каждого контура своя вероятность

    # изображение с контурами данного класса и отдельно картинка с маской вероятности данного класса
    # тогда у меня 4 картинки будет
    
    # это вроде для диффузии что-то 
    def some_visualize(
        self, images, true_masks, pred_masks, class_names_dict, colors, epoch=None
    ):
        true_masks = true_masks.detach().cpu().numpy()
        pred_masks = pred_masks.detach().cpu().numpy()

        for i in range(len(images)):
            image = images[i][0].cpu().numpy()
            image = (image * 255).astype(np.uint8)
            image = cv2.cvtColor(
                image, cv2.COLOR_GRAY2BGR
            )  # Преобразование в трехканальный формат для рисования цветных контуров

            combined_images = []

            for class_idx in range(pred_masks[i].shape[0]):
                prob_mask = pred_masks[i][class_idx]
                # Применение пороговой фильтрации
                thresholded_mask = self.auto_thresh(prob_mask)

                prob_heatmap = (prob_mask * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(prob_heatmap, cv2.COLORMAP_JET)

                # Добавить надпись с классом для тепловой карты
                class_name = class_names_dict.get(
                    class_idx + 1, f"Class {class_idx + 1}"
                )
                cv2.putText(
                    heatmap,
                    class_name,
                    (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                combined_images.append(heatmap)

                # Рисование контуров на исходном изображении
                contours_image = image.copy()
                cv2.putText(
                    contours_image,
                    class_name,
                    (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )  # Название класса на контурной карте
                contours, _ = cv2.findContours(
                    (thresholded_mask * 255).astype(np.uint8),
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if (
                        area > 100
                    ):  # Условие для отображения средних вероятностей только для крупных областей
                        # Создание маски для текущего контура
                        mask_contour = np.zeros_like(prob_mask)
                        cv2.drawContours(
                            mask_contour, [contour], -1, 1, thickness=cv2.FILLED
                        )
                        # Сумма вероятностей в области контура
                        sum_probabilities = np.sum(prob_mask * mask_contour)
                        # Средняя вероятность в области контура
                        avg_probability = sum_probabilities / area

                        cv2.drawContours(contours_image, [contour], -1, (255, 0, 0), 1)
                        # Добавить надпись с вероятностью и классом
                        text = f"{avg_probability:.4f}"
                        text_size = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1
                        )[0]
                        text_x = contour[0][0][0]
                        text_y = contour[0][0][1]

                        # Избегать выхода текста за границы изображения
                        if text_x + text_size[0] > image.shape[1]:
                            text_x = image.shape[1] - text_size[0]
                        if text_y - text_size[1] < 0:
                            text_y = text_size[1]

                        cv2.putText(
                            contours_image,
                            text,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.5,
                            (0, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )

                combined_images.append(contours_image)

            # Организация изображений в квадрат
            num_images = len(combined_images)
            grid_size = int(np.ceil(np.sqrt(num_images)))
            grid_image = np.zeros(
                (grid_size * image.shape[0], grid_size * image.shape[1], 3),
                dtype=np.uint8,
            )

            for idx, img in enumerate(combined_images):
                row = idx // grid_size
                col = idx % grid_size
                grid_image[
                    row * image.shape[0] : (row + 1) * image.shape[0],
                    col * image.shape[1] : (col + 1) * image.shape[1],
                    :,
                ] = img

            if epoch is None:
                cv2.imwrite(
                    f"{self.output_path}/image_{self.image_counter}.jpg", grid_image
                )
            else:
                cv2.imwrite(
                    f"{self.output_path}/epoch_{epoch}_image_{self.image_counter}.jpg",
                    grid_image,
                )

            self.image_counter += 1

    
    def visualize_diffusion(self, images, initial_noise, class_names_dict, colors, num_iterations=10):
        images = images.detach().cpu().numpy()
        initial_noise = initial_noise.detach().cpu().numpy()
        # print("initial_noise shape", initial_noise.shape)

        for i in range(len(images)):
            image = images[i][0]#.cpu().numpy()
            image = (image * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            combined_images = []
            current_noise = initial_noise[i]
            # print("current_noise shaep", current_noise.shape)

            for iteration in range(num_iterations):
                prob_mask = current_noise[1] ##############
                
                max_value = np.max(prob_mask)  # Находим максимальное значение в маске
                # print(f"Максимальное значение на итерации {iteration + 1}: {max_value}")


                thresholded_mask = (prob_mask * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(thresholded_mask, cv2.COLORMAP_JET)

                class_name = f"Iteration {iteration + 1}"
                cv2.putText(
                    heatmap,
                    class_name,
                    (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                combined_images.append(heatmap)

                contours_image = image.copy()
                cv2.putText(
                    contours_image,
                    class_name,
                    (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                contours, _ = cv2.findContours(
                    thresholded_mask,
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:
                        mask_contour = np.zeros_like(prob_mask)
                        cv2.drawContours(
                            mask_contour, [contour], -1, 1, thickness=cv2.FILLED
                        )
                        sum_probabilities = np.sum(prob_mask * mask_contour)
                        avg_probability = sum_probabilities / area

                        cv2.drawContours(contours_image, [contour], -1, (255, 0, 0), 1)
                        text = f"{avg_probability:.4f}"
                        text_size = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1
                        )[0]
                        text_x = contour[0][0][0]
                        text_y = contour[0][0][1]

                        if text_x + text_size[0] > image.shape[1]:
                            text_x = image.shape[1] - text_size[0]
                        if text_y - text_size[1] < 0:
                            text_y = text_size[1]

                        cv2.putText(
                            contours_image,
                            text,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.5,
                            (0, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )

                combined_images.append(contours_image)

            num_images = len(combined_images)
            grid_size = int(np.ceil(np.sqrt(num_images)))
            grid_image = np.zeros(
                (grid_size * image.shape[0], grid_size * image.shape[1], 3),
                dtype=np.uint8,
            )

            for idx, img in enumerate(combined_images):
                row = idx // grid_size
                col = idx % grid_size
                grid_image[
                    row * image.shape[0] : (row + 1) * image.shape[0],
                    col * image.shape[1] : (col + 1) * image.shape[1],
                    :,
                ] = img

            cv2.imwrite(
                f"{self.output_path}/image_{self.image_counter}.jpg", grid_image
            )

            self.image_counter += 1