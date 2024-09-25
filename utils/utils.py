import csv
import os
from typing import Any, List, Optional, Tuple

import torch


def save_best_metrics_to_csv(best_metrics, csv_file):
    # Проверка, существует ли CSV файл
    file_exists = os.path.isfile(csv_file)

    # Если файл существует, считываем данные
    if file_exists:
        with open(csv_file, mode="r", newline="") as file:
            reader = csv.reader(file)
            data = list(reader)
        headers = data[0]
        rows = data[1:]
    else:
        # Если файл не существует, инициализируем заголовки и строки
        headers = (
            ["experiment", "epoch", "train_loss", "val_loss"]
            + [
                "val_iou_class_" + str(i)
                for i in range(best_metrics["val_metrics"]["IOU"].size(0))
            ]
            + [
                "val_f1_class_" + str(i)
                for i in range(best_metrics["val_metrics"]["F1"].size(0))
            ]
            + [
                "val_area_probs_f1_class_" + str(i)
                for i in range(best_metrics["val_metrics"]["area_probs_F1"].size(0))
            ]
            + ["val_mean_iou", "val_mean_f1", "val_mean_area_probs_f1"]
        )
        rows = []

    val_iou = [round(v.item(), 2) for v in best_metrics["val_metrics"]["IOU"]]
    val_f1 = [round(v.item(), 2) for v in best_metrics["val_metrics"]["F1"]]
    val_area_probs_f1 = [
        round(v.item(), 2) for v in best_metrics["val_metrics"]["area_probs_F1"]
    ]

    # Создаем строку с данными метрик
    row = (
        [
            best_metrics["experiment"],
            best_metrics["epoch"],
            round(best_metrics["train_loss"], 2),
            round(best_metrics["val_loss"], 2),
        ]
        # + best_metrics["val_metrics"]["IOU"].tolist()
        # + best_metrics["val_metrics"]["F1"].tolist()
        # + best_metrics["val_metrics"]["area_probs_F1"].tolist()
        + val_iou
        + val_f1
        + val_area_probs_f1
        + [
            round(best_metrics["val_metrics"]["IOU"].mean().item(), 2),
            round(best_metrics["val_metrics"]["F1"].mean().item(), 2),
            round(best_metrics["val_metrics"]["area_probs_F1"].mean().item(), 2),
        ]
    )

    # Проверяем, есть ли запись для текущего эксперимента, и обновляем ее
    experiment_exists = False
    for i, existing_row in enumerate(rows):
        if existing_row[0] == best_metrics["experiment"]:
            rows[i] = row
            experiment_exists = True
            break

    # Если записи для текущего эксперимента нет, добавляем новую строку
    if not experiment_exists:
        rows.append(row)

    # Записываем обновленные данные обратно в CSV файл
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def iou_metric(outputs: torch.Tensor, labels: torch.Tensor, num_classes: int):
    ious = torch.zeros(num_classes)

    for class_idx in range(num_classes):
        binary_outputs = (outputs[:, class_idx, :, :] > 0.5).byte()
        binary_labels = (labels[:, class_idx, :, :]).byte()

        intersection = (binary_outputs & binary_labels).sum((-1, -2))
        union = (binary_outputs | binary_labels).sum((-1, -2))

        iou = (intersection) / (union + 1e-8)
        ious[class_idx] = iou.mean()

    return ious
