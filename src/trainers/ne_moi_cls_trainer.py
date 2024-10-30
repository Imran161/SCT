# класс трейнера

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


def create_experiment_dir(config)
  t = str(time.ctime())
   t = t.replace(' ', '__')
    t = t.replace(':', '_')
    exp_name = "{}_".format(t) + config["exp_name"]
    command = "mkdir {}/{}/models".format(config["runs_path"], exp_name)
    os.system(command)


class Trainer:

    def __init__(self,
                 net,
                 train_dataloader,
                 val_dataloader,
                 config):

        self.net = net
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config

        self.variables = {} вносить всякие переменные
        self.function = {} а сюда функции

        self.variables["phase"] = None

        self.function["val_loss"] = Combo_loss(self.config["val_loss_parameters"])
        self.function["train_loss"] = Combo_loss(self.config["train_loss_parameters"])
        self.function["val_metrics"] = Get_metrics(self.config["val_metrics_parameters"])
        self.function["train_metrics"] = Get_metrics(self.config["train_metrics_parameters"])
        self.function["transforms"] = Get_transforms(self.config["transform_parameters"])

    def load_best_model(self):
        checkpoint = "{}{}/models/best.pth".format(self.path, self.exp_name)
        self.net.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.net.to(self.device)

    def load_best_weight(self):
        self.loss_class.pixel_weight = self.best_loss_weight

    def reset_epoch_counters(self):

        self.variables["actual_epoch_loss"] = 0.0
        self.variables["actual_num_of_step"] = 0.0

    def step(self):

        self.optimizer.zero_grad()

        self.variables["masks_pred"] = self.net(self.variables["images"])

        if self.variables["phase"] == "train":

            loss = self.train_loss(self.variables["masks_pred"],
                                   self.variables["true_masks"])
            loss.backward()
            self.optimizer.step()

        if self.variables["phase"] == "val":

            loss = self.val_loss(self.variables["masks_pred"],
                                 self.variables["true_masks"])

        self.variables["actual_loss"] = loss.item()
        self.variables["actual_masks_pred"] = masks_pred.detach()
        self.variables["actual_epoch_loss"] += self.loss
        self.variables["actual_num_of_step"] += 1

    def activate_save_manager(self):

        optimal_target_metrics = True
        best_loss = True

        for cl in self.config["classes"]
          for key, value in self.variables["actual_epoch_metrics"]:
               if value[cl] <= self.config["target_metrics"][key][cl]:
                    optimal_target_metrics

    def start_epoch(self, DataLoader):

        if self.config["test_mode"] == True:
            self.variables["target_num_of_step"] = config["num_of_step_in_test_mode"]:
        else:
            self.variables["target_num_of_step"] = len(DataLoader)

        with tqdm(total=len(DataLoader), desc=f'Epoch {self.variables["epoch"] + 1}/{self.config["epochs"]}',
                  unit='img') as pbar:

            while self.variables["actual_num_of_step"] <= self.variables["target_num_of_step"]

               batch = DataLoader
                images, true_masks = batch["images"], batch["masks"]

                if self.config["Delete_background"] == True:
                    true_masks = true_masks[:, 1:, :, :].clone()
                if self.function["transform"] != None and self.variables["phase"] == "train":
                    images, true_masks = self.transform(images, true_masks)

                true_masks = true_masks.to(self.device, dtype=torch.float32)
                images = images.to(self.device, dtype=torch.float32)

                self.variables["images"], self.variables["true_masks"] = images, true_masks

                self.step(images, true_masks)

                pbar.set_postfix(**{'loss (batch)': self.variables["actual_loss"]})
                pbar.update(1)

                with torch.no_grad():
                    self.metrics.update_counter(self.variables["true_masks"].detach(),
                                                self.variables["actual_masks_pred"].detach())

        self.variables["actual_epoch_metrics"] = self.metrics.calculate_metrics()
        self.variables["epoch_mean_loss"] = self.variables["actual_epoch_loss"]/self.variables["actual_num_of_step"]

    def start_train_cycle(self):

        self.variables["best_epoch_loss"] = 1000000
        self.variables["counter"] = self.config["early_stop"]
        self.variables["easy_mode"] = False

        for epoch in range(self.config["epohs"]):

            self.variables["epoch"] = epoch
            self.variables["phase"] == "train"
            self.net.train()
            self.start_epoch(DataLoader=self.train_dataloader)

            self.variables["phase"] == "val"
            self.net.eval()

            with torch.no_grad():
                self.start_epoch(DataLoader=self.val_dataloader)

            self.print_metrics()
            self.save_metrics()
            self.activate_save_manager()
            self.activate_write_manager()
            self.activate_special_parameter_manager()


#             self.print_metrics(epoch, phase, train_metrics)
#             self.save_metrics(epoch, phase, train_metrics) как это лучше реализовать?
#             print("Текущие веса классов:")
#             print(self.loss_class.pixel_weight)
#                 self.print_metrics(epoch, phase, metrics)
#                 self.save_metrics(epoch, phase, metrics)

              optimality = self.condition_check(metrics)

               if optimality == True:
                    print(f"На эпохе {str(epoch+1)} достигнуты оптимальные метрики")
                    torch.save(self.net.state_dict(), "{}{}/models/optimal_model.pth".format(self.path, self.exp_name))

                mean_area_probs_AUROC = self.calculate_impotrant_classes_mean_AUROCs(metrics)

                if mean_area_probs_AUROC != None:                
                    if mean_area_probs_AUROC > best_mean_probs_AUROC:
                        print(
                            f"На эпохе {str(epoch+1)} достигнут новый лучший mean_confidence_AUROC:{mean_area_probs_AUROC}")
                        torch.save(self.net.state_dict(),
                                   "{}{}/models/best_area_prob_AUROC.pth".format(self.path, self.exp_name))
                        best_mean_probs_AUROC = mean_area_probs_AUROC

                if val_epoch_loss < best_loss:
                    print("На эпохе {} достигнута новая лучшая ошибка валидации, которая составляет {}".format(str(epoch+1),
                                                                                                               str(val_epoch_loss)))
                    best_loss = val_epoch_loss

                    self.best_loss_weight = self.loss_class.pixel_weight

                    counter = self.counter
                    torch.save(self.net.state_dict(), "{}{}/models/best.pth".format(self.path, self.exp_name))
                else:
                    counter -= 1

                self.writer.add_scalar("Val_epoch_loss", val_epoch_loss, epoch)
                self.writer.add_scalar("Best_loss", best_loss, epoch)
                torch.save(self.net.state_dict(), "{}{}/models/last.pth".format(self.path, self.exp_name))

                if self.epochs/(epoch+1) > 2 and easy_mode==False:
                    if self.loss_class.pixel_weight != None and self.Optim_pw == True:
                        self.opt_pixel_weight(train_metrics)


#             if counter==0:
#                 if easy_mode==False:
#                     self.load_best_model()
#                     self.load_best_weight()
#                     easy_mode = True
#                     print(f"Коррекция весов остановлена на эпохе {epoch}. Загружены лучшие веса")
#                     counter = self.counter
#                 else:
#                     print(f"Обучение остановлено на эпохе {epoch}. Ошибка не улучшается")
#                     break
