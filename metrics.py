import numpy as np
import torch
import cv2
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import warnings


class Detection_metrics:
    
    def __init__(self, mode, num_classes, treshold = 0.5):
        self.mode = mode
        self.num_classes = num_classes
        self.treshold = treshold
        self.IOU = np.zeros(self.num_classes,)
        self.tp = np.zeros(self.num_classes,)
        self.fp = np.zeros(self.num_classes,)
        self.fn = np.zeros(self.num_classes,)
 
        self.Average_IOU = np.zeros(self.num_classes,).astype(np.float32)
        self.Average_tp = np.zeros(self.num_classes,).astype(np.float32)
        self.Average_fp = np.zeros(self.num_classes,).astype(np.float32)
        self.Average_fn = np.zeros(self.num_classes,).astype(np.float32)
        
        self.all_confidences = []
        self.all_probs = []
        self.all_true_label = []
    

    def calculate_confidence(self, prob_masks):
        smooth = 0.00001
        list_of_class_confidences = []
        
        for i in range(prob_masks.size()[0]):
            loss = (prob_masks[i] * torch.log(prob_masks[i]+smooth)).sum()/(prob_masks[i].sum()+smooth)
            loss+=((1 - prob_masks[i]) * torch.log(1 - prob_masks[i]+smooth)).sum()/((1 - prob_masks[i]).sum()+smooth)
            loss/=2
            confidence = loss.exp()
            confidence = confidence.item()
            # print("confidence", confidence) # нанов нет
            list_of_class_confidences.append(confidence)
       
        return list_of_class_confidences
    
    def calculate_probs(self, prob_masks):   
        list_of_class_probs = []
        tr = 0.3
        for i in range(prob_masks.size()[0]):
            class_mask = prob_masks[i]
            class_mask[class_mask<tr] = 0
            non_zero = torch.count_nonzero(class_mask)
            
            if non_zero!=0:
                prob_sum = class_mask.sum()
                prob = prob_sum/(non_zero)
                prob = prob.item()
                list_of_class_probs.append(prob)
            else:
                list_of_class_probs.append(0)
                
        return list_of_class_probs 
    
    def calculate_confidences_and_probs(self, true_mask, pred_mask):
        
        for example in range(pred_mask.size()[0]):
            confidence_list = self.calculate_confidence(pred_mask[example])
            # print("confidence_list", confidence_list) # вроде нанов нет
            probs_list = self.calculate_probs(pred_mask[example])
            # print("probs_list", probs_list) # # вроде нанов нет
            self.all_confidences.append(confidence_list)
            self.all_probs.append(probs_list)
      
            true_label = self.true_mask_to_true_label(true_mask[example])
            self.all_true_label.append(true_label)
            
    def calculate_AUROC(self, true_label, probs):
        
        classes_AUROC = []
        classes_recall = []
        classes_precession = []
        classes_F1 = []
        
        # print("true_label", true_label)
        # print("len true_label", len(true_label))
        # print("probs", probs) # тут наны есть 
        # print("probs", len(probs))
        
        class_nums = len(true_label[0])
        
        for class_num in range(class_nums):
            new_prob = []
            new_label = []
            for label, prob in zip(true_label, probs):
                new_label.append(label[class_num])
                new_prob.append(prob[class_num])
            
            # print("new_label", new_label)
            # print("new_prob", new_prob) # и тут соответсвенно
            
            # try:    
            
            # попробую нулем Nan заменить
            # new_prob = np.nan_to_num(new_prob, nan=0.0)
            # if np.isnan(np.sum(new_label)):
            #     print("NaN detected in new_label")
            #     continue
            
            # if np.isnan(np.sum(new_prob)):
            #     print("NaN detected in new_prob")
            #     continue

            fpr, tpr, thresholds = roc_curve(new_label, new_prob)
            roc_auc = auc(fpr, tpr)
            # print("roc_auc", roc_auc)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            confusion_matrix = np.zeros((2, 2))
            
            for i in range(len(new_label)):
                if new_prob[i] >= optimal_threshold:
                    predicted = 1
                else:
                    predicted = 0
                confusion_matrix[int(new_label[i]), predicted] += 1


            # вот тут у меня была ошибка потому что не попадались картинки с патологиями, поэтому обработаю исключение 
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                if confusion_matrix[1, 1] + confusion_matrix[1, 0] == 0:
                    recall = 0
                    precession = 0
                else:
                    recall = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0])
                    precession  = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
                    
            # а вот тут деление на 0 было 
            if recall + precession == 0:
                F1 = 0
            else:
                F1 = 2*(recall*precession)/(recall+precession)
            
            if not np.isnan(roc_auc):
                classes_AUROC.append(roc_auc)
                # print("classes_AUROC", classes_AUROC)
            classes_recall.append(recall)
            classes_precession.append(precession)
            classes_F1.append(F1)
            # except:
            #     # pass
            #     if sum(new_label) == 0 or sum(new_label) == len(new_label):
            #         classes_AUROC.append(1.0)
      
            #     else:
            #         # Если в данных отсутствуют положительные или отрицательные примеры
            #         classes_AUROC.append(0.0)
      

        return classes_AUROC, classes_recall, classes_precession, classes_F1
                
    def mask2contours(self, mask):
        contours, h = cv2.findContours(mask.astype(int).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        out = []
        if len(contours)!=0:
            s = cv2.contourArea(contours[0])
            if s!=0:
                for c in contours:
                    if cv2.contourArea(c)/s>0.02:
                        out.append(c)    
        return out
    
    def true_mask_to_true_label(self, true_mask):
        
        true_labels = []  
        
        for i in range(true_mask.size()[0]):
            true_labels.append(true_mask[i].max().item())

        return true_labels

    def use_treshold(self, pred_mask):
#         print(pred_mask)
        pred_mask[pred_mask>self.treshold] = 1
        pred_mask[pred_mask<=self.treshold] = 0
        return pred_mask
    
    def use_argmax(self, pred_mask):
        ncl = pred_mask.size()[1]
        for i in range(np.shape(pred_mask)[0]):
            index_mask = torch.argmax(pred_mask[i], dim=0)
            cl_pred_mask = F.one_hot(index_mask, ncl).permute(2, 0, 1).long()
            pred_mask[i] = cl_pred_mask
        return pred_mask
    
    def to_numpy (self, mask):
        mask = mask.to('cpu').detach().numpy()
        return mask

    def contours2mask(self, size, contours):
        mask = np.zeros(size)
        mask = cv2.drawContours(mask, contours, -1, 1, -1)
        return mask
     
    def numpy_IOU(self, mask1, mask2):
        intersection = np.sum(mask1 * mask2)
        if intersection == 0:
            return 0.0
        union = np.sum(np.logical_or(mask1, mask2).astype(int))
        return intersection / union
    
    def update_counter(self, true_mask, pred_mask):
        
        batch_size = true_mask.size()[0]
        true_mask = true_mask.detach()
        pred_mask = pred_mask.detach()
        
        self.calculate_confidences_and_probs(true_mask, pred_mask)
        
        if self.mode == "ML":
            pred_mask = self.use_treshold(pred_mask)
        elif self.mode == "MC":
            pred_mask = self.use_argmax(pred_mask)
        elif self.mode == "modern":
            
            pred_mask[:, 0, :,:] = self.use_treshold(pred_mask[:, 0, :,:])
            pred_mask[:,1:,:,:] = self.use_argmax(pred_mask[:,1:,:,:])

            for example in range(pred_mask.size()[0]):
                for image_class in range(pred_mask.size()[1]):
                    if image_class!=0:
                        pred_mask[example,image_class,:,:][true_mask[example,0,:,:]==0]=0
          
        true_mask = self.to_numpy(true_mask)
        pred_mask = self.to_numpy(pred_mask)
        
        batch_IOU = np.zeros(self.num_classes,)
        batch_tp = np.zeros(self.num_classes,)
        batch_fp = np.zeros(self.num_classes,)
        batch_fn = np.zeros(self.num_classes,)
        
        Average_batch_IOU = np.zeros(self.num_classes,).astype(np.float32)
        Average_batch_tp = np.zeros(self.num_classes,).astype(np.float32)
        Average_batch_fp = np.zeros(self.num_classes,).astype(np.float32)
        Average_batch_fn = np.zeros(self.num_classes,).astype(np.float32)
       
        
        for i in range(batch_size):
            for j in range(self.num_classes):
                instance_IOU, instance_tp, instance_fp, instance_fn = self.easy_detect_objects(true_mask[i,j], pred_mask[i, j])
                # a_instance_IOU, a_instance_tp, a_instance_fp, a_instance_fn = self.detect_objects(true_mask[i,j], pred_mask[i, j])
                
                batch_IOU[j]+=instance_IOU
                batch_tp[j]+=instance_tp
                batch_fp[j]+=instance_fp
                batch_fn[j]+=instance_fn
                
                # Average_batch_IOU[j]+=a_instance_IOU
                # Average_batch_tp[j]+=a_instance_tp
                # Average_batch_fp[j]+=a_instance_fp
                # Average_batch_fn[j]+=a_instance_fn
                
        self.IOU+=batch_IOU
        self.tp +=batch_tp
        self.fp +=batch_fp
        self.fn +=batch_fn  
        
        # self.Average_IOU +=Average_batch_IOU
        # self.Average_tp +=Average_batch_tp
        # self.Average_fp +=Average_batch_fp
        # self.Average_fn +=Average_batch_fn
        

    def easy_detect_objects(self, true_mask, pred_mask):
        
        ################################################################################## вот тут надо метрику лучше сделать 
        
        instance_tp = 0
        instance_fp = 0
        instance_fn = 0
        
        true_label = np.max(true_mask)
        
        if true_label==0:
            instance_IOU = 0
            if np.max(pred_mask)==0:
                pred_label = 0
            else:
                pred_label = 1
                
        else:
            intersection = true_mask * pred_mask
            intersection = np.sum(intersection)
            detect_sum = intersection/np.sum(true_mask)
            if detect_sum>0.5:
                pred_label = 1
                union = np.sum(np.logical_or(true_mask, pred_mask).astype(int))
                instance_IOU =  intersection / union    
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
        
        # убрал а то не работает
        # try:
        area_probs_AUROC, area_probs_recall, area_probs_precession, area_probs_F1 = self.calculate_AUROC(self.all_true_label, self.all_confidences)
        
        confidence_AUROC, confidence_recall, confidence_precession, confidence_F1 = self.calculate_AUROC(self.all_true_label, self.all_probs)


        self.all_confidences = []
        self.all_probs = []
        self.all_true_label = []
        
        # тут вот так сделал чтобы на 0 не делилось
        with np.errstate(divide='ignore', invalid='ignore'):
            recall = np.divide(self.tp, self.tp + self.fn)
            recall[np.isnan(recall)] = 0  # Заменяем NaN значения на ноль

        with np.errstate(divide='ignore', invalid='ignore'):
            precession = np.divide(self.tp, self.tp + self.fp)
            precession[np.isnan(precession)] = 0  # Заменяем NaN значения на ноль
            recall = np.divide(self.tp, self.tp + self.fn)
            recall[np.isnan(recall)] = 0  # Заменяем NaN значения на ноль
            F1 = 2 * (recall * precession) / (recall + precession)
            F1[np.isnan(F1)] = 0  # Заменяем NaN значения на ноль
            IOU = np.divide(self.IOU, self.tp)
            IOU[np.isnan(IOU)] = 0  # Заменяем NaN значения на ноль

        # а было вот так
        # recall = self.tp/(self.tp+self.fn)
        # precession = self.tp/(self.tp+self.fp)
        # F1 = 2*(recall*precession)/(recall+precession)
        # IOU = self.IOU/self.tp
        
        
        
        self.IOU = np.zeros(self.num_classes,)
        self.tp = np.zeros(self.num_classes,)
        self.fp = np.zeros(self.num_classes,)
        self.fn = np.zeros(self.num_classes,)

        metrics = {}
        
        metrics["IOU"] = torch.tensor(IOU)
        # print("IOU", IOU) # IOU [0.53183877 0.         0.         0.        ]
        metrics["recall"] = torch.tensor(recall)
        metrics["precession"] = torch.tensor(precession)
        metrics["F1"] = torch.tensor(F1)

        metrics["confidence_AUROC"] = torch.tensor(confidence_AUROC)    # и он
        
        metrics["confidence_recall"] = torch.tensor(confidence_recall)
        metrics["confidence_precession"] = torch.tensor(confidence_precession)
        metrics["confidence_F1"] = torch.tensor(confidence_F1)
        
        
        
        metrics["area_probs_AUROC"] = torch.tensor(area_probs_AUROC)    # он не работает
        metrics["area_probs_recall"] = torch.tensor(area_probs_recall)
        metrics["area_probs_precession"] = torch.tensor(area_probs_precession)
        metrics["area_probs_F1"] = torch.tensor(area_probs_F1)
    
        
#         metrics["Average_IOU"] = torch.tensor(Average_IOU)
#         metrics["Average_recall"] = torch.tensor(Average_recall)
#         metrics["Average_precession"] = torch.tensor(Average_precession)
#         metrics["Average_F1"] = torch.tensor(Average_F1)
        
        return metrics
    
        # except:
        #     pass
    

