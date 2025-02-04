import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve


# тут конфиг дикт тоже на вход сделать
# потом класс этот разделить надо на несколько


class DetectionMetrics:
    def __init__(self, mode: str, num_classes: int, threshold=0.5):
        self.mode = mode
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset_metrics()

    def reset_metrics(self):
        self.IOU = np.zeros(self.num_classes)
        self.tp = np.zeros(self.num_classes)
        self.fp = np.zeros(self.num_classes)
        self.fn = np.zeros(self.num_classes)

        self.advanced_IOU = np.zeros(self.num_classes)
        self.advanced_tp = np.zeros(self.num_classes)
        self.advanced_fp = np.zeros(self.num_classes)
        self.advanced_fn = np.zeros(self.num_classes)

        self.all_confidences = []
        self.all_probs = []
        self.all_true_labels = []

    def calc_confidence(self, prob_masks):
        smooth = 1e-5
        class_confidences = []

        for i in range(prob_masks.size(0)):
            mask = prob_masks[i]
            loss = (mask * torch.log(mask + smooth)).sum() / (mask.sum() + smooth)
            loss += ((1 - mask) * torch.log(1 - mask + smooth)).sum() / (
                (1 - mask).sum() + smooth
            )
            confidence = (loss / 2).exp().item()
            class_confidences.append(confidence)

        return class_confidences

    def calc_probs(self, prob_masks):
        class_probs = []
        tr = 0.3
        for i in range(prob_masks.size(0)):
            mask = prob_masks[i]
            mask[mask < tr] = 0
            non_zero = torch.count_nonzero(mask)

            if non_zero != 0:
                prob = (mask.sum() / non_zero).item()
                class_probs.append(prob)
            else:
                class_probs.append(0)

        return class_probs

    def calc_confidences_and_probs(self, true_mask, pred_mask):
        for example in range(pred_mask.size(0)):
            confidence_list = self.calc_confidence(pred_mask[example])
            probs_list = self.calc_probs(pred_mask[example])
            self.all_confidences.append(confidence_list)
            self.all_probs.append(probs_list)

            true_label = self.true_mask_to_true_label(true_mask[example])
            self.all_true_labels.append(true_label)

    def calc_AUROC(self, true_labels, probs):
        classes_AUROC = []
        classes_recall = []
        classes_precision = []
        classes_F1 = []

        class_nums = len(true_labels[0])

        for class_num in range(class_nums):
            new_labels = [label[class_num] for label in true_labels]
            new_probs = [prob[class_num] for prob in probs]

            fpr, tpr, thresholds = roc_curve(new_labels, new_probs)
            roc_auc = auc(fpr, tpr)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]

            confusion_matrix = np.zeros((2, 2))

            for i in range(len(new_labels)):
                if new_probs[i] >= optimal_threshold:
                    predicted = 1
                else:
                    predicted = 0
                confusion_matrix[int(new_labels[i]), predicted] += 1

            if (confusion_matrix[1, 1] + confusion_matrix[1, 0]) != 0:
                recall = confusion_matrix[1, 1] / (
                    confusion_matrix[1, 1] + confusion_matrix[1, 0]
                )
            else:
                recall = 0

            if (confusion_matrix[0, 0] + confusion_matrix[0, 1]) != 0:
                precision = confusion_matrix[0, 0] / (
                    confusion_matrix[0, 0] + confusion_matrix[0, 1]
                )
            else:
                precision = 0

            if (recall + precision) != 0:
                F1 = 2 * (recall * precision) / (recall + precision)
            else:
                F1 = 0

            if not np.isnan(roc_auc):
                classes_AUROC.append(roc_auc)
            classes_recall.append(recall)
            classes_precision.append(precision)
            classes_F1.append(F1)

        return classes_AUROC, classes_recall, classes_precision, classes_F1

    def true_mask_to_true_label(self, true_mask):
        return [true_mask[i].max().item() for i in range(true_mask.size(0))]

    def use_threshold(self, pred_mask):
        pred_mask[pred_mask > self.threshold] = 1
        pred_mask[pred_mask <= self.threshold] = 0
        return pred_mask

    def use_argmax(self, pred_mask):
        ncl = pred_mask.size(1)
        for i in range(pred_mask.size(0)):
            index_mask = torch.argmax(pred_mask[i], dim=0)
            pred_mask[i] = F.one_hot(index_mask, ncl).permute(2, 0, 1).long()
        return pred_mask

    def update_counter(self, true_mask, pred_mask):  # , advanced_metrics=False):
        batch_size = true_mask.size(0)
        true_mask = true_mask.detach()
        pred_mask = pred_mask.detach()

        self.calc_confidences_and_probs(true_mask, pred_mask)

        #########################
        # вынести в отдельный метод, чтобы отдельно вызывать в трейне постпроцессинг этот
        if self.mode == "ML":
            pred_mask = self.use_threshold(pred_mask)
        elif self.mode == "MC":
            pred_mask = self.use_argmax(pred_mask)
        elif self.mode == "modern":
            pred_mask[:, 0, :, :] = self.use_threshold(pred_mask[:, 0, :, :])
            pred_mask[:, 1:, :, :] = self.use_argmax(pred_mask[:, 1:, :, :])

            for example in range(pred_mask.size(0)):
                for image_class in range(pred_mask.size(1)):
                    if image_class != 0:
                        pred_mask[example, image_class, :, :][
                            true_mask[example, 0, :, :] == 0
                        ] = 0
        ##########################

        true_mask = true_mask.cpu().detach().numpy()
        pred_mask = pred_mask.cpu().detach().numpy()

        batch_IOU = np.zeros(self.num_classes)
        batch_tp = np.zeros(self.num_classes)
        batch_fp = np.zeros(self.num_classes)
        batch_fn = np.zeros(self.num_classes)

        advanced_batch_IOU = np.zeros(self.num_classes)
        advanced_batch_tp = np.zeros(self.num_classes)
        advanced_batch_fp = np.zeros(self.num_classes)
        advanced_batch_fn = np.zeros(self.num_classes)

        for i in range(batch_size):
            for j in range(self.num_classes):
                (
                    instance_IOU,
                    instance_tp,
                    instance_fp,
                    instance_fn,
                ) = self.calc_basic_metrics(true_mask[i, j], pred_mask[i, j])

                batch_IOU[j] += instance_IOU
                batch_tp[j] += instance_tp
                batch_fp[j] += instance_fp
                batch_fn[j] += instance_fn

                (
                    advanced_instance_IOU,
                    advanced_instance_tp,
                    advanced_instance_fp,
                    advanced_instance_fn,
                ) = self.calc_advanced_metrics(true_mask[i, j], pred_mask[i, j])

                advanced_batch_IOU[j] += advanced_instance_IOU
                advanced_batch_tp[j] += advanced_instance_tp
                advanced_batch_fp[j] += advanced_instance_fp
                advanced_batch_fn[j] += advanced_instance_fn

        self.IOU += batch_IOU
        self.tp += batch_tp
        self.fp += batch_fp
        self.fn += batch_fn

        self.advanced_IOU += advanced_batch_IOU
        self.advanced_tp += advanced_batch_tp
        self.advanced_fp += advanced_batch_fp
        self.advanced_fn += advanced_batch_fn

    def calc_fp(self, true_mask, pred_mask):
        true_label = np.max(true_mask)
        if true_label == 0:
            return 1

        false_positive_area = pred_mask * (1 - true_mask)
        # Площадь истинной маски и разности
        area_true = np.sum(true_mask)
        area_fp = np.sum(false_positive_area)

        if area_true == 0:
            return 1
        fp = np.ceil(area_fp / area_true)

        return int(fp)

    def calc_advanced_metrics(self, true_mask, pred_mask):
        instance_tp, instance_fp, instance_fn, instance_IOU = 0, 0, 0, 0
        true_label = np.max(true_mask)

        if true_label == 0:  # Нет объектов в истинной маске
            if np.max(pred_mask) != 0:  # Ложное срабатывание
                instance_fp += self.calc_fp(true_mask, pred_mask)
        else:
            intersection = np.sum(true_mask * pred_mask)
            union = np.sum(np.logical_or(true_mask, pred_mask))

            if intersection > 0:
                detect_sum = intersection / np.sum(true_mask)
                if detect_sum > 0.5:
                    instance_tp += 1
                    instance_IOU = intersection / union
                else:
                    instance_fn += 1
                    instance_fp += self.calc_fp(true_mask, pred_mask)
            else:  # Пропущенный объект
                instance_fn += 1
                instance_fp += self.calc_fp(true_mask, pred_mask)

        return instance_IOU, instance_tp, instance_fp, instance_fn

    def calc_basic_metrics(self, true_mask, pred_mask):
        instance_tp, instance_fp, instance_fn, instance_IOU = 0, 0, 0, 0
        true_label = np.max(true_mask)

        if true_label == 0:
            if np.max(pred_mask) != 0:
                instance_fp = 1
        else:
            intersection = np.sum(true_mask * pred_mask)
            union = np.sum(np.logical_or(true_mask, pred_mask).astype(int))

            if intersection > 0:
                detect_sum = intersection / np.sum(true_mask)
                if detect_sum > 0.5:
                    instance_tp = 1
                    instance_IOU = intersection / union
                else:
                    instance_fn = 1
                    instance_fp = 1
            else:
                instance_fn = 1
                instance_fp = 1

        return instance_IOU, instance_tp, instance_fp, instance_fn

    def calc_metrics(self):
        (
            area_probs_AUROC,
            area_probs_recall,
            area_probs_precision,
            area_probs_F1,
        ) = self.calc_AUROC(self.all_true_labels, self.all_confidences)
        (
            confidence_AUROC,
            confidence_recall,
            confidence_precision,
            confidence_F1,
        ) = self.calc_AUROC(self.all_true_labels, self.all_probs)

        with np.errstate(divide="ignore", invalid="ignore"):
            recall = np.nan_to_num(self.tp / (self.tp + self.fn))
            precision = np.nan_to_num(self.tp / (self.tp + self.fp))
            F1 = np.nan_to_num(2 * (recall * precision) / (recall + precision))
            IOU = np.nan_to_num(self.IOU / self.tp)

            advanced_recall = np.nan_to_num(
                self.advanced_tp / (self.advanced_tp + self.advanced_fn)
            )
            advanced_precision = np.nan_to_num(
                self.advanced_tp / (self.advanced_tp + self.advanced_fp)
            )
            advanced_F1 = np.nan_to_num(
                2
                * (advanced_recall * advanced_precision)
                / (advanced_recall + advanced_precision)
            )
            advanced_IOU = np.nan_to_num(self.advanced_IOU / self.advanced_tp)

        self.reset_metrics()

        metrics = {
            "IOU": torch.tensor(IOU),
            "recall": torch.tensor(recall),
            "precision": torch.tensor(precision),
            "F1": torch.tensor(F1),
            "confidence_AUROC": torch.tensor(confidence_AUROC),
            "confidence_recall": torch.tensor(confidence_recall),
            "confidence_precision": torch.tensor(confidence_precision),
            "confidence_F1": torch.tensor(confidence_F1),
            "area_probs_AUROC": torch.tensor(area_probs_AUROC),
            "area_probs_recall": torch.tensor(area_probs_recall),
            "area_probs_precision": torch.tensor(area_probs_precision),
            "area_probs_F1": torch.tensor(area_probs_F1),
            "advanced_IOU": torch.tensor(advanced_IOU),
            "advanced_recall": torch.tensor(advanced_recall),
            "advanced_precision": torch.tensor(advanced_precision),
            "advanced_F1": torch.tensor(advanced_F1),
        }

        return metrics
