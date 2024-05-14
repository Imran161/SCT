import numpy as np
import torch
import torch.nn.functional as F

class MetricsCalculator:
    def __init__(self, num_classes, threshold=0.5, mode="modern"):
        self.num_classes = num_classes
        self.threshold = threshold
        self.mode = mode
        
        # Инициализация переменных для метрик
        self.IOU = np.zeros(self.num_classes)
        self.tp = np.zeros(self.num_classes)
        self.fp = np.zeros(self.num_classes)
        self.fn = np.zeros(self.num_classes)
        # self.Average_IOU = np.zeros(self.num_classes).astype(np.float32)
        # self.Average_tp = np.zeros(self.num_classes).astype(np.float32)
        # self.Average_fp = np.zeros(self.num_classes).astype(np.float32)
        # self.Average_fn = np.zeros(self.num_classes).astype(np.float32)
    
    def use_threshold(self, pred_mask):
        pred_mask[pred_mask > self.threshold] = 1
        pred_mask[pred_mask <= self.threshold] = 0
        return pred_mask

    def use_argmax(self, pred_mask):
        ncl = pred_mask.size()[1]
        for i in range(np.shape(pred_mask)[0]):
            index_mask = torch.argmax(pred_mask[i], dim=0)
            cl_pred_mask = F.one_hot(index_mask, ncl).permute(2, 0, 1).long()
            pred_mask[i] = cl_pred_mask
        return pred_mask

    def update_counter(self, true_mask, pred_mask):
        batch_size = true_mask.size()[0]
        true_mask = true_mask.detach()
        pred_mask = pred_mask.detach()

        # self.calculate_confidences_and_probs(true_mask, pred_mask)

        if self.mode == "ML":
            pred_mask = self.use_threshold(pred_mask)
        elif self.mode == "MC":
            pred_mask = self.use_argmax(pred_mask)
        elif self.mode == "modern":
            pred_mask[:, 0, :, :] = self.use_threshold(pred_mask[:, 0, :, :])
            pred_mask[:, 1:, :, :] = self.use_argmax(pred_mask[:, 1:, :, :])

            for example in range(pred_mask.size()[0]):
                for image_class in range(pred_mask.size()[1]):
                    if image_class != 0:
                        pred_mask[example, image_class, :, :][true_mask[example, 0, :, :] == 0] = 0

        true_mask = self.to_numpy(true_mask)
        pred_mask = self.to_numpy(pred_mask)

        for i in range(batch_size):
            for j in range(self.num_classes):
                instance_IOU, instance_tp, instance_fp, instance_fn = self.easy_detect_objects(true_mask[i, j],
                                                                                                 pred_mask[i, j])
                # a_instance_IOU, a_instance_tp, a_instance_fp, a_instance_fn = self.detect_objects(true_mask[i, j],
                #                                                                                    pred_mask[i, j])

                self.IOU[j] += instance_IOU
                self.tp[j] += instance_tp
                self.fp[j] += instance_fp
                self.fn[j] += instance_fn

                # self.Average_IOU[j] += a_instance_IOU
                # self.Average_tp[j] += a_instance_tp
                # self.Average_fp[j] += a_instance_fp
                # self.Average_fn[j] += a_instance_fn

    def easy_detect_objects(self, true_mask, pred_mask):
        instance_tp = 0
        instance_fp = 0
        instance_fn = 0

        true_label = np.max(true_mask)

        if true_label == 0:
            instance_IOU = 0
            if np.max(pred_mask) == 0:
                pred_label = 0
            else:
                pred_label = 1

        else:
            intersection = true_mask * pred_mask
            intersection = np.sum(intersection)
            detect_sum = intersection / np.sum(true_mask)
            if detect_sum > 0.5:
                pred_label = 1
                union = np.sum(np.logical_or(true_mask, pred_mask).astype(int))
                instance_IOU = intersection / union
            else:
                pred_label = 0
                instance_IOU = 0

        if true_label == 1 and pred_label == 1:
            instance_tp = 1
        if true_label == 1 and pred_label == 0:
            instance_fn = 1
        if true_label == 0 and pred_label == 1:
            instance_fp = 1

        return instance_IOU, instance_tp, instance_fp, instance_fn


  
    def calculate_metrics(self):
        
        area_probs_AUROC, area_probs_recall, area_probs_precession, area_probs_F1 = self.calculate_AUROC(self.all_true_label, self.all_confidences)
        
        confidence_AUROC, confidence_recall, confidence_precession, confidence_F1 = self.calculate_AUROC(self.all_true_label, self.all_probs)

        self.all_confidences = []
        self.all_probs = []
        self.all_true_label = []
        
        recall = self.tp/(self.tp+self.fn)
        precession = self.tp/(self.tp+self.fp)
        F1 = 2*(recall*precession)/(recall+precession)
        IOU = self.IOU/self.tp
        
        self.IOU = np.zeros(self.num_classes,)
        self.tp = np.zeros(self.num_classes,)
        self.fp = np.zeros(self.num_classes,)
        self.fn = np.zeros(self.num_classes,)
        
        metrics = {}
        
        metrics["IOU"] = torch.tensor(IOU)
        metrics["recall"] = torch.tensor(recall)
        metrics["precession"] = torch.tensor(precession)
        metrics["F1"] = torch.tensor(F1)

        metrics["confidence_AUROC"] = torch.tensor(confidence_AUROC)
        
        metrics["confidence_recall"] = torch.tensor(confidence_recall)
        metrics["confidence_precession"] = torch.tensor(confidence_precession)
        metrics["confidence_F1"] = torch.tensor(confidence_F1)
        
        
        
        metrics["area_probs_AUROC"] = torch.tensor(area_probs_AUROC)
        metrics["area_probs_recall"] = torch.tensor(area_probs_recall)
        metrics["area_probs_precession"] = torch.tensor(area_probs_precession)
        metrics["area_probs_F1"] = torch.tensor(area_probs_F1)
       
        return metrics


    def to_numpy(self, tensor):
        return tensor.cpu().numpy()
