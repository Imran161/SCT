import cv2
import skimage.io as io
import json
import random 
import matplotlib
import pycocotools
from cv2 import INTER_NEAREST
from pycocotools.coco import COCO
import codecs
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch

import torch.nn.functional as F
import segmentation_models_pytorch as smp
import time
from torch import nn, optim
import torchmetrics
from tqdm import tqdm
from torchvision import transforms
import pickle
from sklearn.metrics import roc_curve, auc
from metrics import Detection_metrics

BCELoss = torch.nn.BCELoss(reduction='none')
KLDivLoss = torch.nn.KLDivLoss(reduction='none')
CosineSimilarity = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

FocalLoss_softmax = smp.losses.FocalLoss(mode = "multiclass")
FocalLoss_sigmoid = smp.losses.FocalLoss(mode = "multilabel")


class nemoi_Detection_metrics:
    
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
            probs_list = self.calculate_probs(pred_mask[example])
            self.all_confidences.append(confidence_list)
            self.all_probs.append(probs_list)
      
            true_label = self.true_mask_to_true_label(true_mask[example])
            self.all_true_label.append(true_label)
            
    def calculate_AUROC(self, true_label, probs):
        
        classes_AUROC = []
        classes_recall = []
        classes_precession = []
        classes_F1 = []
        
        class_nums = len(true_label[0])
        
        for class_num in range(class_nums):
            new_prob = []
            new_label = []
            for label, prob in zip(true_label, probs):
                new_label.append(label[class_num])
                new_prob.append(prob[class_num])
                
            fpr, tpr, thresholds = roc_curve(new_label, new_prob)
            roc_auc = auc(fpr, tpr)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            confusion_matrix = np.zeros((2, 2))
            
            for i in range(len(new_label)):
                if new_prob[i] >= optimal_threshold:
                    predicted = 1
                else:
                    predicted = 0
                confusion_matrix[int(new_label[i]), predicted] += 1

            recall = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0])
            precession  = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
            F1 = 2*(recall*precession)/(recall+precession)
            
            classes_AUROC.append(roc_auc)
            classes_recall.append(recall)
            classes_precession.append(precession)
            classes_F1.append(F1)

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
                a_instance_IOU, a_instance_tp, a_instance_fp, a_instance_fn = self.detect_objects(true_mask[i,j], pred_mask[i, j])
                
                batch_IOU[j]+=instance_IOU
                batch_tp[j]+=instance_tp
                batch_fp[j]+=instance_fp
                batch_fn[j]+=instance_fn
                
                Average_batch_IOU[j]+=a_instance_IOU
                Average_batch_tp[j]+=a_instance_tp
                Average_batch_fp[j]+=a_instance_fp
                Average_batch_fn[j]+=a_instance_fn
                
        self.IOU+=batch_IOU
        self.tp +=batch_tp
        self.fp +=batch_fp
        self.fn +=batch_fn  
        
        self.Average_IOU +=Average_batch_IOU
        self.Average_tp +=Average_batch_tp
        self.Average_fp +=Average_batch_fp
        self.Average_fn +=Average_batch_fn
        

    def easy_detect_objects(self, true_mask, pred_mask):
        
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
        
    
    def detect_objects(self, true_mask, pred_mask):
        return np.nan, np.nan, np.nan, np.nan
            
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


class Trainer:
    
    def __init__(self, net, train_dataloader, val_dataloader, 
                 classes, Mode, device,
                 loss, learning_rate = 0.0001, epochs=120, exp_name="unnamed_experiment",
                 path = r"runs/",
                 transform = None, Delete_background = False,
                 test_mode = False,
                 b = None,
                 early_stop = None,
                 important_classes = [],
                 target_metrick = None):
        
        t = str(time.ctime())
        t = t.replace(' ', '__')
        t = t.replace(':', '_')
        self.exp_name = "{}_".format(t)+exp_name
        self.path = path
        
        command = "mkdir {}{}/models".format(self.path, self.exp_name)

        self.net = net
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.classes = classes
        self.Mode = Mode
        self.Delete_background = Delete_background
        self.test_mode = test_mode
        
        if early_stop!=None:
            self.counter = early_stop
        else:
            self.counter = epochs
        
        if self.Delete_background==True:
            self.classes.remove("фон")
        if self.Delete_background==True and self.Mode == "MC":
            print("При использовании мультиклассовой сегментации фон должен быть!")
        self.learning_rate = learning_rate
        self.device = device
        self.loss_class = loss
        self.loss = self.loss_class.calculate_loss
        self.epochs = epochs
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=self.learning_rate, weight_decay=1e-8, momentum=0.9)
        self.net.to(self.device)
        self.writer = SummaryWriter(r"runs/{}".format(self.exp_name))
        print(command)
        os.system(command)
        self.transform = transform
        self.metrics = Detection_metrics(Mode, len(classes))
        self.b = b
        self.best_loss_weight = None
        self.important_classes = important_classes
        self.target_metrick = target_metrick
        self.mean_optimal_metrick = 0
    
    def load_best_model(self):
        checkpoint = "{}{}/models/best.pth".format(self.path, self.exp_name)
        self.net.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.net.to(self.device)  

    def load_best_weight(self):
        self.loss_class.pixel_weight = self.best_loss_weight
        
    
    def mask2label(self, mask):
        label = torch.amax(mask, [-1,-2])
        return label
    
    def opt_pixel_weight(self, metrics):
        
        recall = metrics["recall"]
        precession = metrics["precession"]
        F1Score = metrics["F1"]
        
        b = self.b
        
        if b==None:
            b = 1

        for image_class, cl_name in enumerate(self.classes):
            neg_coef = 1
            pos_coef = 1
            
            if recall[image_class].item()!=0 and precession[image_class].item()!=0:
                
                neg_coef = (1/b)*recall[image_class].item()/F1Score[image_class].item()
                pos_coef = (b)*precession[image_class].item()/F1Score[image_class].item()

                xsd = recall[image_class].item()/precession[image_class].item()
                if xsd>0.9 and xsd<1.1:
                    neg_coef = 1
                    pos_coef = 1
                class_coef = pos_coef
            else:
                pos_coef = 2.0
                class_coef = 2.0
                neg_coef = 0.5
                
                    
            self.loss_class.pixel_weight[0, image_class]*=pos_coef
            self.loss_class.pixel_weight[1, image_class]*=neg_coef
            self.loss_class.pixel_weight[2, image_class]*=class_coef
            

    def IOU(self, mask1, mask2):
        intersection = (mask1 * mask2).sum()
        if intersection == 0:
            return 0.0
        union = torch.logical_or(mask1, mask2).to(torch.int).sum()
        return intersection / union
    
    def print_metrics(self, epoch, phase, metrics): 

        print("Завершилась эпоха номер {}".format(str(epoch+1)))
        print("Фаза: {}".format(phase))

        for n, cl in enumerate(self.classes):
            for key, value in metrics.items():
                print(f"Для класса - {cl}, показатель {str(key)} составляет {str(metrics[key][n].item())}")
        

    def save_metrics(self, epoch, phase, metrics):
        
        for n, cl in enumerate(self.classes):
            for key, value in metrics.items():
                self.writer.add_scalar(f"{phase} {key} {cl}", metrics[key][n].item(), epoch+1)

    def calculate_impotrant_classes_mean_AUROCs(self, metrics):     
        
        area_probs_AUROC = 0
        count = 0
        
        for n, cl in enumerate(self.classes):
            if cl in self.important_classes:
                area_probs_AUROC+=metrics["area_probs_AUROC"][n].item()
                count+=1
        
        if count!=0:
            mean_area_probs_AUROC = area_probs_AUROC/count
            return mean_area_probs_AUROC
        else:
            return None
    
    def condition_check(self, metrics):
                    
        mean_optimal_metrick = 0
        optimality = True

        for count, cl in enumerate(self.classes):
            if cl in self.important_classes:
                print(f"Для класса {cl}:")
                for key, value in self.target_metrick.items():
                    m = metrics[key][count].item()
                    print(f"Целевое значение метрики {key} = {value}, реальное значение = {m}")
                    if m<=value:
                        print(f"Значение метрики {key} не оптимально")
                        optimality=False
                    else:
                        print(f"Значение метрики {key} достаточно")
                        
                    mean_optimal_metrick+=m
        
        if optimality==True:
            print(f"Старая сумма целевых метрик {self.mean_optimal_metrick}")
            print(f"Новая сумма целевых метрик {mean_optimal_metrick}")
            if mean_optimal_metrick<=self.mean_optimal_metrick:
                print("Обновление отклонено")
                optimality = False
            else:
                print("Обновление принято")
                self.mean_optimal_metrick = mean_optimal_metrick
                optimality = True
        else:
            print("Метрики не оптимальны")
                
        return optimality
    
    def IOU_for_all_classes (self, masks_pred, true_mask):
        masks_pred = masks_pred.long()
        true_mask = true_mask.long()
        out = torch.zeros(masks_pred.size()[1])
        out = out.to(self.device)
        for i in range(masks_pred.size()[0]):
            for j in range(masks_pred.size()[1]):
                iou = self.IOU(masks_pred[i,j,:,:], true_mask[i,j,:,:])
                out[j] += iou
        return out
    
    def val_segm_loss(self, masks_pred, true_masks):
        
        if self.Mode != "modern":

            if self.Mode == "ML":
                masks_pred = F.sigmoid(masks_pred)
            if self.Mode == "MC":
                masks_pred = F.softmax(masks_pred, dim = 1)


            smooth=0.00001       

            intersection = (true_masks * masks_pred)
            intersection = intersection.sum([-1,-2])

            fps = (masks_pred * (1. - true_masks)).sum([-1,-2])
            fns = ((1. - masks_pred) * true_masks).sum([-1,-2])

            loss = 1-(intersection + smooth)/(fps+fns+intersection+smooth) 

            for example in range(loss.size()[0]):
                for image_class in range(loss.size()[1]):
                    if true_masks[example, image_class].max()==0:
                        loss[example, image_class]*=0

            loss_mean = torch.mean(loss)
        
        else:
            
            masks_pred = my_activation_function(masks_pred)
            
            masks_pred_detect = masks_pred[:,0,:,:]
            true_masks_detect = true_masks[:,0,:,:]
            
            masks_pred_classification = masks_pred[:,1:,:,:]
            true_masks_classification = true_masks[:,1:,:,:]
            
            smooth=0.00001       

            intersection = (masks_pred_detect * true_masks_detect)
            intersection = intersection.sum([-1,-2])

            fps = (masks_pred_detect * (1. - true_masks_detect)).sum([-1,-2])
            fns = ((1. - masks_pred_detect) * true_masks_detect).sum([-1,-2])

            loss_detect = 1-(intersection + smooth)/(fps+fns+intersection+smooth)
            
            count = 0
            for example in range(loss_detect.size()[0]):
                if true_masks_classification[example, 0].max()==0:
                    loss_detect[example]*=0
                else:
                    count+=1
            
            loss_detect = torch.sum(loss_detect)
            
            if count!=0:
                loss_detect = loss_detect/count
                        
            for example in range(masks_pred_classification.size()[0]):
                for image_class in range(masks_pred_classification.size()[1]):
                    masks_pred_classification[example, image_class, :, :][true_masks_detect[example]==0] = 0
            
            
            
            smooth=0.00001       

            intersection = (masks_pred_classification * true_masks_classification)
            intersection = intersection.sum([-1,-2])

            fps = (masks_pred_classification * (1. - true_masks_classification)).sum([-1,-2])
            fns = ((1. - masks_pred_classification) * true_masks_classification).sum([-1,-2])

            loss_classification = 1-(intersection + smooth)/(fps+fns+intersection+smooth)
            
            count = 0
            for example in range(loss_classification.size()[0]):
                for image_class in range(loss_classification.size()[1]):
                    if true_masks_classification[example, image_class].max()==0:
                        loss_classification[example, image_class]*=0
                    else:
                        count+=1
            
            
            loss_classification = torch.sum(loss_classification)
            
            if count!=0:            
                loss_classification = loss_classification/count
            

            loss_mean = (loss_detect + loss_classification)/2
            
        return loss_mean
        
    
    def step(self, images, true_masks, phase):
        
        self.optimizer.zero_grad()
        masks_pred = self.net(images)
        
        if phase!="val":
            loss = self.loss(masks_pred, true_masks)
            loss.backward()
            self.optimizer.step()
        else:
            loss = self.val_segm_loss(masks_pred, true_masks)
        
        return loss.item(), masks_pred
    
    def start_epoch(self, epoch, phase, DataLoader):
    
        epoch_loss = 0
        step = 0
     
        with tqdm(total=len(DataLoader), desc=f'Epoch {epoch + 1}/{self.epochs}',
                  unit='img') as pbar:
            
            if self.test_mode==True:
                t=0
            
            for batch in DataLoader:
                
                images = batch["images"]
                true_masks = batch ["masks"]
                
                if self.Mode == "MC":
                    size = true_masks.size()[1]
                    true_masks = torch.argmax(true_masks, dim=1)
                    true_masks = F.one_hot(true_masks, size).permute(0, 3, 1, 2).long()
                                    
                if self.Delete_background==True:
                    true_masks = true_masks[:,1:,:,:].clone().detach()
                
                if self.transform!=None and phase=="train":
                    images, true_masks = self.transform(images, true_masks)
                
                true_masks = true_masks.to(self.device, dtype=torch.float32)
                images = images.to(self.device, dtype=torch.float32)
                
                loss, masks_pred = self.step(images, true_masks, phase)
                epoch_loss += loss
                step+=1
                pbar.set_postfix(**{'loss (batch)': loss})
                pbar.update(1)
                
                with torch.no_grad():
                    masks_pred = masks_pred.detach()
                    true_masks = true_masks.detach()
                    #new metrics
                    if self.Mode == "ML":
                        masks_pred = F.sigmoid(masks_pred)
                    if self.Mode == "MC":
                        masks_pred = F.softmax(masks_pred, dim = 1)
                    if self.Mode == "modern":
                        masks_pred = my_activation_function(masks_pred)

                    self.metrics.update_counter(true_masks, masks_pred)

                if self.test_mode==True:
                    t+=1
                    if t==20:
                        break                
        
        metrics = self.metrics.calculate_metrics()
        return metrics, epoch_loss/step
        
    def start_train_cycle(self):
        best_loss = 1000000
        best_mean_confidence_AUROC = 0
        best_mean_probs_AUROC = 0
        counter = self.counter
        easy_mode = False
        for epoch in range(self.epochs):
            if counter==0:
                if easy_mode==False:
                    self.load_best_model()
                    self.load_best_weight()
                    easy_mode = True
                    print(f"Коррекция весов остановлена на эпохе {epoch}. Загружены лучшие веса")
                    counter = self.counter
                else:
                    print(f"Обучение остановлено на эпохе {epoch}. Ошибка не улучшается")
                    break
                    
            phase = "train"
            self.net.train()
            train_metrics, epoch_loss = self.start_epoch(epoch, phase, DataLoader = self.train_dataloader)
            self.print_metrics(epoch, phase, train_metrics)
            self.save_metrics(epoch, phase, train_metrics)
            
            print("Текущие веса классов:")
            print(self.loss_class.pixel_weight)
            
            phase = "val"
            self.net.eval()
            with torch.no_grad():
                
                metrics, val_epoch_loss = self.start_epoch(epoch, phase, DataLoader = self.val_dataloader)
                self.print_metrics(epoch, phase, metrics)
                self.save_metrics(epoch, phase, metrics)
                
                optimality = self.condition_check(metrics)
                        
                if optimality == True:
                    print(f"На эпохе {str(epoch+1)} достигнуты оптимальные метрики")
                    torch.save(self.net.state_dict(), "{}{}/models/optimal_model.pth".format(self.path, self.exp_name))
                
                mean_area_probs_AUROC = self.calculate_impotrant_classes_mean_AUROCs(metrics)
                
                if mean_area_probs_AUROC!=None:                
                    if mean_area_probs_AUROC>best_mean_probs_AUROC:
                        print(f"На эпохе {str(epoch+1)} достигнут новый лучший mean_confidence_AUROC:{mean_area_probs_AUROC}")
                        torch.save(self.net.state_dict(), "{}{}/models/best_area_prob_AUROC.pth".format(self.path, self.exp_name))
                        best_mean_probs_AUROC = mean_area_probs_AUROC
                
                if val_epoch_loss<best_loss:
                    print("На эпохе {} достигнута новая лучшая ошибка валидации, которая составляет {}".format(str(epoch+1),
                                                                                                               str(val_epoch_loss)))
                    best_loss=val_epoch_loss
                    
                    self.best_loss_weight = self.loss_class.pixel_weight
                    
                    counter = self.counter
                    torch.save(self.net.state_dict(), "{}{}/models/best.pth".format(self.path, self.exp_name))
                else:
                    counter-=1

                self.writer.add_scalar("Val_epoch_loss", val_epoch_loss, epoch)
                self.writer.add_scalar("Best_loss", best_loss, epoch)
                torch.save(self.net.state_dict(), "{}{}/models/last.pth".format(self.path, self.exp_name))
                
                if self.epochs/(epoch+1)>2 and easy_mode==False:
                    if self.loss_class.pixel_weight!=None:
                        self.opt_pixel_weight(train_metrics)