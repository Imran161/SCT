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
import os
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
import hashlib
from imutils import paths

class Universal_json_Segmentation_Dataset():

    def __init__(self,
                 json_file_path,
                 delete_list,
                 base_classes,
                 out_classes, 
                 dataloader=False,
                 resize=None,
                 recalculate = False, 
                 delete_null = False):
        
        # self.recalculate = recalculate
        self.resize = resize
        self.json_file_path = json_file_path
        self.out_cl_hash = self.generate_class_id(out_classes)
        
        if self.json_file_path[-1]!=r"/":
            self.json_file_path+=r"/"
        
        full_json_file_path = self.json_file_path+"annotations/instances_default.json"
        self.coco = COCO(full_json_file_path)
        self.dataloader=dataloader
        self.catIDs = self.coco.getCatIds()
#         self.use_compact_classes = use_compact_classes
        self.ids_idx_dict = {}
        self.delete_list = delete_list
        self.delete_null = delete_null
        self.out_classes = out_classes
        
        self.base_classes = base_classes
        
        self.list_of_name_out_classes = ["фон"]
        self.list_of_name_base_classes = ["фон"]
        
        for segmentation_class in self.base_classes:
                self.list_of_name_base_classes.append(segmentation_class["name"])
        for segmentation_class in self.out_classes:
                self.list_of_name_out_classes.append(segmentation_class["name"])
        
        self.cats_to_classes={}
        cats = self.coco.loadCats(self.catIDs)
        
        for cl in self.base_classes:
            for cat in cats:
                # print("cat[name]", cat["name"])
                # print("type[name]", type(cat["name"]))
                # print("cl[name]", cl["name"])
                # print("type(cl[name])", type(cl["name"]))
                # print(cat["name"]==cl["name"])
                if cl["name"]==cat["name"]:
                    self.cats_to_classes[cat["id"]]=cl["id"]
        
        self.check_create_train_val_list(recalculate)
        self.check_create_weight(recalculate)
                    
          
     
        
        
        self.colors = [((251, 206, 177), 'Абрикосовым'), ((127, 255,  212), 'Аквамариновым'), ((255, 36, 0), 'Алым'), ((153, 102, 204), 'Аметистовым'),
                       ((153, 0, 102), 'Баклажановым'), ((48, 213, 200), 'Бирюзовым'), ((152, 251,  152), 'Бледно зеленым'),
                       ((213, 113, 63), 'Ванильным'), ((100, 149,  237), 'Васильковым'), ((34,139,34), 'Зелёный лесной'), ((0,0,255), 'Синий'),
                       ((75,0,130), 'Индиго'), ((255,0,255), 'Чёрный'), ((0,51,  153), 'Маджента'), ((65,105,225), 'Королевский синий'),
                       ((255,255,0), 'Жёлтый'), ((255,69,0), 'Оранжево-красный'), ((255,0,0), 'Темно синим'), ((0,51,  153), 'Красный'),
                       ((255,215,0), 'Золотой'), ((250,128,114), 'Лососевый'), ((255,99,71), 'Томатный'), ((255,215,0), 'Золотой'),
                       ((0,139,139), 'Тёмный циан'), ((0,255,255), 'Морская волна')]
         
    def getImgIds_all_cats(self, imgIDs, catIDs):
        ids_list = []
        for catID in catIDs:
            ids_list.extend(self.coco.getImgIds(imgIds = imgIDs, catIds=catID))
        return list(set(ids_list))        
 
    def __len__(self):
        return len(self.train_list)+len(self.val_list)
    
    def generate_class_id (self, classes):
        my_hash = 0
        for cl in classes:
            my_str = str(cl["id"])+cl["name"]+cl["name"]+str(cl["summable_masks"])+str(cl['subtractive_masks'])
            my_str = my_str.encode('utf-8')
            my_hash+=int.from_bytes(bytes(hashlib.sha256(my_str).hexdigest(), 'utf-8'), "little")
        return str(my_hash)   
    
    def check_create_train_val_list (self, recalculate):
       
             
        try:
            train_list = open(self.json_file_path + f'train_list_{self.out_cl_hash}.pickle', 'rb')
            val_list = open(self.json_file_path + f'val_list_{self.out_cl_hash}.pickle', 'rb')
            all_img_list = open(self.json_file_path + f'all_img_list_{self.out_cl_hash}.pickle', 'rb')
            
            self.train_list = pickle.load(train_list)
            self.val_list = pickle.load(val_list)
            self.all_img_list = pickle.load(all_img_list)
           
            if recalculate != False:
                raise ValueError('stuff is not in content')
        
        except:
            
            imgIds = self.coco.getImgIds() 
            print("готовые данные не обраружены или включена рекалькуляция")
            train_list = []
            val_list = []
            all_img_list = []
            
            
            for img_index in imgIds:
                anns_ids = self.coco.getAnnIds(imgIds=img_index, catIds=self.catIDs)
                anns = self.coco.loadAnns(anns_ids)
                # print("anns", anns)
                save = True
                if len(anns)==0 and self.delete_null == True:
                    save=False
                    print("отсутствуют какие либо метки:", img_index)
                for ann in anns:
                    cat = self.coco.loadCats(ann["category_id"])[0]
                    if cat["name"] in self.delete_list:
                        save = False
                        non_support_class = cat["name"]
                        print(f"недопустимое значение метки:{non_support_class}", img_index)
                # print("save", save)
                if save:
                    x=random.randint(1, 100)
                    if x>=80:
                        val_list.append(img_index)
                    else:
                        train_list.append(img_index)
                        self.train_list = train_list
                    all_img_list.append(img_index)
            
            self.train_list = train_list
            self.val_list = val_list
            self.all_img_list = all_img_list
            
            with open(self.json_file_path + f'train_list_{self.out_cl_hash}.pickle', 'wb') as train_list:
                pickle.dump(self.train_list, train_list)
            with open(self.json_file_path + f'val_list_{self.out_cl_hash}.pickle', 'wb') as val_list:
                pickle.dump(self.val_list, val_list)
            with open(self.json_file_path + f'all_img_list_{self.out_cl_hash}.pickle', 'wb') as all_img_list:
                pickle.dump(self.all_img_list, all_img_list)
                
    def check_create_weight(self, recalculate):

        try:
            
            TotalTrain = open(self.json_file_path + f'TotalTrain_{self.out_cl_hash}.pickle', 'rb')
            TotalVal = open(self.json_file_path + f'TotalVal_{self.out_cl_hash}.pickle', 'rb')
            pixel_TotalTrain = open(self.json_file_path + f'pixel_TotalTrain_{self.out_cl_hash}.pickle', 'rb')
            pixel_TotalVal = open(self.json_file_path + f'pixel_TotalVal_{self.out_cl_hash}.pickle', 'rb')
            
            self.TotalTrain = pickle.load(TotalTrain)
            self.TotalVal = pickle.load(TotalVal)
            self.pixel_TotalTrain = pickle.load(pixel_TotalTrain)
            self.pixel_TotalVal = pickle.load(pixel_TotalVal)
            
            
            if recalculate != False:
                raise ValueError('stuff is not in content')
        
        except:
            
            print("готовые веса не обраружены или включена рекалькуляция")
        
            noc = len(self.list_of_name_out_classes)
            TotalTrain = np.zeros(noc)
            TotalVal = np.zeros(noc)

            pixel_TotalTrain = np.zeros(noc)
            pixel_TotalVal = np.zeros(noc)
            

            for i in self.train_list:
                result = self.__getitem__ (i)
                # print(type(result))
                # print("result[images]", result["images"])
                image, mask = result["images"], result["masks"]
                for j in range(noc):
                    TotalTrain[j]+=mask[j].max().item()
                    pixel_TotalTrain[j]+=mask[j].sum().item()

            for i in self.val_list:
                result = self.__getitem__ (i)
                image, mask = result["images"], result["masks"]
                for j in range(noc):
                    TotalVal[j]+=mask[j].max().item()
                    pixel_TotalVal+=mask[j].sum().item()

            self.TotalTrain = TotalTrain
            self.TotalVal = TotalVal
            self.pixel_TotalTrain = pixel_TotalTrain
            self.pixel_TotalVal = pixel_TotalVal

            with open(self.json_file_path + f'TotalTrain_{self.out_cl_hash}.pickle', 'wb') as Total:
                pickle.dump(self.TotalTrain, Total)
            with open(self.json_file_path + f'TotalVal_{self.out_cl_hash}.pickle', 'wb') as Total:
                pickle.dump(self.TotalVal, Total)
            with open(self.json_file_path + f'pixel_TotalTrain_{self.out_cl_hash}.pickle', 'wb') as pix_Total:
                pickle.dump(self.pixel_TotalTrain, pix_Total)
            with open(self.json_file_path + f'pixel_TotalVal_{self.out_cl_hash}.pickle', 'wb') as pix_Total:
                pickle.dump(self.pixel_TotalVal, pix_Total)

    def show_me_contours(self, idx):
        gray_image, mask, rgb_image = self.__getitem__ (idx, contures=True) 
        ######
        # print("mask", mask)
        # print("type(mask)", type(mask)) # <class 'numpy.ndarray'>
        # print("mask.shape", mask.shape) # (6, 512, 512)           # 6 классов
        # values, counts = np.unique(mask, return_counts=True)
        # for v, c in zip(values, counts):
        #     print(f"v:{v}, c:{c}")
        ########### это one hot маска тут всегда 0 и 1 будут просто 1 на том слое какого класса картинка ################
        #######
        # а в тетрадке вот так
        # v:0, c:262081
        # v:1, c:63
        #######
        # print(mask[:1,:,:].sum())
        plt.rcParams["figure.figsize"] = [12, 12]
        plt.rcParams["figure.autolayout"] = True
        k = 0
        for i in range(np.shape(mask)[0]):
            # надо куда-то поворот маски потом кстати еще добавить
            # надо findContours на image_from_polygon делать
            # то есть я у себя в коде сначала делал картинку по полигону а потом на ней искал контуры
            
            #######
            # мой код
            # image = np.zeros(image_shape, dtype=np.uint8)
    
            # polygon = np.array(polygon).reshape((-1, 2)).astype(np.int32)
            # cv2.fillPoly(image, [polygon], (255, 255, 255))
            #######
            
            # plt.imshow(mask[i])
            # plt.show()
            
            contours, h = cv2.findContours(mask[i].astype(int).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # print("contours", contours)
            rgb_image = cv2.drawContours(rgb_image, contours, -1, (self.colors[i][0][0], self.colors[i][0][1], self.colors[i][0][2]), 2)
            if np.max(mask[i])==1 and i!=0:
                text = self.list_of_name_out_classes[i]+ " " +str(np.max(mask[i]))
                # для 255 снимка такой вывод
                # print("i", i) # i 3
                # print("list_of_name_out_classes", self.list_of_name_out_classes) # ['фон', '1', '2', '3', '4', '5']
                # print("list_of_name_out_classes[i]", self.list_of_name_out_classes[i]) # 3
                # print("text", text) # 1 выводит
                plt.text(2000, k, text , color = (self.colors[i][0][0]/255, self.colors[i][0][1]/255, self.colors[i][0][2]/255)) 
                # так попробую
                cv2.putText(rgb_image, f"label class: {text}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA) 
                k+=50
        
        # сохраню фотки, сделаю папки по классам
        # если патологии нет, то ошибка будет, потому что text не существует
        # print("text", text[0])
        if not os.path.exists("/home/imran-nasyrov/sct_project/sct_data/output_images/" + text[0]):
            os.mkdir("/home/imran-nasyrov/sct_project/sct_data/output_images/" + text[0])  
                
        all_files = get_all_files("/home/imran-nasyrov/sct_project/sct_data/output_images/" + text[0])
        if len(all_files) <= 5:
            cv2.imwrite(f"/home/imran-nasyrov/sct_project/sct_data/output_images/{text[0]}/output_image_{idx}.jpg", rgb_image)
        
        
        # plt.imshow(rgb_image)
        # plt.show()
    
    def to_out_classes(self, mask):
        size = np.shape(mask)

        new_mask = np.zeros((len(self.out_classes)+1,size[1],size[2]))
        new_mask[0]=1
     
        for out_class in self.out_classes:
            for i in out_class["summable_masks"]:
                new_mask[out_class["id"],:,:][mask[i,:,:]==1]=1
            for i in out_class["subtractive_masks"]:
                new_mask[out_class["id"],:,:][mask[i,:,:]==1]=0
            new_mask[0][new_mask[out_class["id"],:,:]==1]=0  
        
        return new_mask

    def __getitem__(self, idx, contures=False):
        
        # print("idx", idx)
        images_description = self.coco.loadImgs(idx)[0]
        # print("images_description", images_description)
        image_path = self.json_file_path+"images/"+images_description['file_name']
        # print("image_path", image_path)
        rgb_image = cv2.imread(image_path)
        # print("rgb_image", rgb_image.shape)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        
        print("self.catIDs", self.catIDs)
        print("idx", idx)
        anns_ids = self.coco.getAnnIds(imgIds=idx, catIds=self.catIDs, iscrowd=None)
        print("anns_ids", anns_ids)
        anns = self.coco.loadAnns(anns_ids)
        print("anns", anns)
        # для чего вот эта размерность? len(self.catIDs)-len(self.delete_list)+1 это кол-во классов
        mask = np.zeros((len(self.catIDs)-len(self.delete_list)+1, int(images_description["height"]), int(images_description["width"])))
        
        height = int(images_description["height"])
        width = int(images_description["width"])
        #######
        # print("после np.zeros")
        # values, counts = np.unique(mask, return_counts=True)
        # for v, c in zip(values, counts):
        #     print(f"v:{v}, c:{c}")
        #v:0.0, c:1835008
        # print("mask in getitem shape", mask.shape) # (7, 512, 512)
        #######
        
        bboxes = []
        
        for ann in anns:  
            # print("ann", ann)
            # вот эту строку убрал она странная какая-то
            cat = self.coco.loadCats(ann["category_id"])[0]
            # cat = ann["category_id"] # и сделал так, все заработало, только пока нарисовался диком но без маски
            
            # тут вот этот if добавил, без него ошибка была
            # if len(self.delete_list) > 0:
            # print("cat[name]", cat)
            if cat["name"] not in self.delete_list:
                class_mask = self.coco.annToMask(ann)
                class_idx = self.cats_to_classes[ann["category_id"]]
                mask[class_idx][class_mask==1]=1
                
                bbox = ann["bbox"]
                bboxes.append(bbox)
                
        # print("до to_out_classes")
        # values, counts = np.unique(mask, return_counts=True)
        # for v, c in zip(values, counts):
        #     print(f"v:{v}, c:{c}")
            
        mask = self.to_out_classes(mask)
        ########
        # print("после to_out_classes")
        # values, counts = np.unique(mask, return_counts=True)
        # for v, c in zip(values, counts):
        #     print(f"v:{v}, c:{c}")
        # v:0.0, c:1310720
        # v:1.0, c:262144
        #######
            
        # print("resize", self.resize)
        # print(contures)
        if self.resize is not None and contures==False :
            # print("мы в этом ифе")
            image = torch.unsqueeze(torch.tensor(gray_image), 0)
            image = torchvision.transforms.functional.resize(image, (self.resize))
            mask = torchvision.transforms.functional.resize(torch.tensor(mask), (self.resize))
            if self.dataloader==False:
                
                image = torch.unsqueeze(image, 0)
                image = (image-image.min())/(image.max()-image.min()+0.00000001)
                mask = torch.unsqueeze(mask, 0)
                rgb_image = cv2.resize(rgb_image, (self.resize))
                image = image.float()
                mask = mask.long()
                return image, mask, rgb_image  
            
            image = (image-image.min())/(image.max()-image.min()+0.00000001)
            image = image.float()
            mask = mask.long()
            result = {}
            result["images"] = image
            result["masks"] = mask
            result["labels"] = torch.amax(mask, dim=(-1, -2))
            result["values"] = torch.sum(mask, (-1,-2))
            result["bboxes"] = bboxes
            result["rgb_image"] = rgb_image #########
            result["height"] = height
            result["width"] = width
            
            return result
        else:
            # этот иф вызывается для индекса картики который хочу нарисовать, для остальных верхний вызывается
            # ну это потому что в шоу контрс getitem вызывается от contures=True так что все норм
            # print("мы внизу")
            return gray_image, mask, rgb_image # , anns ######################################### тут anns не было

        
class Universal_npz_Segmentation_Dataset():

    def __init__(self,
                 files_path,
                 delete_list,
                 base_classes,
                 out_classes, 
                 dataloader=False,
                 resize=None,
                 recalculate = False):
        
        self.resize = resize
        self.files_path = files_path
        
        self.out_cl_hash = self.generate_class_id(out_classes)
        
        if self.files_path[-1]!=r"/":
            self.files_path+=r"/"
            
        self.images_paths = self.get_all_files(f"{self.files_path}images")
        self.masks_paths = self.get_all_files(f"{self.files_path}masks")
        self.all_images_and_mask_paths = None
       
        os.system(f"mkdir {self.files_path}hash")
        self.dataloader=dataloader
       
        self.delete_list = delete_list
        
        self.out_classes = out_classes
        
        self.base_classes = base_classes
        
        self.list_of_name_out_classes = ["фон"]
        self.list_of_name_base_classes = ["фон"]
        
        for segmentation_class in self.base_classes:
                self.list_of_name_base_classes.append(segmentation_class["name"])
        
        for segmentation_class in self.out_classes:
                self.list_of_name_out_classes.append(segmentation_class["name"])
        
        self.check_create_data(recalculate)
        self.check_create_train_val_list(recalculate)
        self.check_create_weight(recalculate)

    def __len__(self):
        return len(self.train_list)+len(self.val_list)
    
    def generate_class_id (self, classes):
        my_hash = 0
        for cl in classes:
            my_str = str(cl["id"])+cl["name"]+cl["name"]+str(cl["summable_masks"])+str(cl['subtractive_masks'])
            my_str = my_str.encode('utf-8')
            my_hash+=int.from_bytes(bytes(hashlib.sha256(my_str).hexdigest(), 'utf-8'), "little")
        return str(my_hash)   
    
    def get_all_files(self, directory):
    
        all_files = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)

        return all_files
    
    def check_create_train_val_list (self, recalculate):
        
        try:
            train_list = open(f"{self.files_path}hash/train_list_{self.out_cl_hash}.pickle", 'rb')
            val_list = open(f"{self.files_path}hash/val_list_{self.out_cl_hash}.pickle", 'rb')
            all_img_list = open(f"{self.files_path}hash/all_img_list_{self.out_cl_hash}.pickle", 'rb')
 
            self.train_list = pickle.load(train_list)
            self.val_list = pickle.load(val_list)
            self.all_img_list = pickle.load(all_img_list)
            print(self.out_cl_hash, recalculate)

            if recalculate != False:
                raise ValueError('stuff is not in content')
        
        except:
            num_imgs = len(self.all_images_and_mask_paths)
            print("Готовые данные не обраружены или включена рекалькуляция")
            train_list = []
            val_list = []
            all_img_list = []
            
            for img_index in range(num_imgs):
                all_img_list.append(img_index)
                x=random.randint(1, 100)
                if x>=80:
                    val_list.append(img_index)
                else:
                    train_list.append(img_index)
            
            self.train_list = train_list
            self.val_list = val_list
            self.all_img_list = all_img_list
            
            with open(f"{self.files_path}hash/train_list_{self.out_cl_hash}.pickle", 'wb') as train_list:
                pickle.dump(self.train_list, train_list)
            with open(f"{self.files_path}hash/val_list_{self.out_cl_hash}.pickle", 'wb') as val_list:
                pickle.dump(self.val_list, val_list)
            with open(f"{self.files_path}hash/all_img_list_{self.out_cl_hash}.pickle", 'wb') as all_img_list:
                pickle.dump(self.all_img_list, all_img_list)
                
    def check_create_weight(self, recalculate):

        try:
            
            TotalTrain = open(f"{self.files_path}hash/TotalTrain_{self.out_cl_hash}.pickle", 'rb')
            TotalVal = open(f"{self.files_path}hash/TotalVal_{self.out_cl_hash}.pickle", 'rb')
            pixel_TotalTrain = open(f"{self.files_path}hash/pixel_TotalTrain_{self.out_cl_hash}.pickle", 'rb')
            pixel_TotalVal = open(f"{self.files_path}hash/pixel_TotalVal_{self.out_cl_hash}.pickle", 'rb')
            
            self.TotalTrain = pickle.load(TotalTrain)
            self.TotalVal = pickle.load(TotalVal)
            self.pixel_TotalTrain = pickle.load(pixel_TotalTrain)
            self.pixel_TotalVal = pickle.load(pixel_TotalVal)
            print(recalculate)
            if recalculate != False:
                raise ValueError('stuff is not in content')
        
        except:
            
            print("готовые веса не обраружены или включена рекалькуляция")
        
            noc = len(self.list_of_name_out_classes)
            TotalTrain = np.zeros(noc)
            TotalVal = np.zeros(noc)

            pixel_TotalTrain = np.zeros(noc)
            pixel_TotalVal = np.zeros(noc)
            

            for i in self.train_list:
                image, mask = self.__getitem__ (i, for_inner_use = True)
                for j in range(noc):
                    TotalTrain[j]+=mask[j].max().item()
                    pixel_TotalTrain[j]+=mask[j].sum().item()

            for i in self.val_list:
                image, mask = self.__getitem__ (i, for_inner_use = True)
                for j in range(noc):
                    TotalVal[j]+=mask[j].max().item()
                    pixel_TotalVal+=mask[j].sum().item()

            self.TotalTrain = TotalTrain
            self.TotalVal = TotalVal
            self.pixel_TotalTrain = pixel_TotalTrain
            self.pixel_TotalVal = pixel_TotalVal
                                              
            with open(f"{self.files_path}hash/TotalTrain_{self.out_cl_hash}.pickle", 'wb') as Total:
                pickle.dump(self.TotalTrain, Total)
            with open(f"{self.files_path}hash/TotalVal_{self.out_cl_hash}.pickle", 'wb') as Total:
                pickle.dump(self.TotalVal, Total)
            with open(f"{self.files_path}hash/pixel_TotalTrain_{self.out_cl_hash}.pickle", 'wb') as pix_Total:
                pickle.dump(self.pixel_TotalTrain, pix_Total)
            with open(f"{self.files_path}hash/pixel_TotalVal_{self.out_cl_hash}.pickle", 'wb') as pix_Total:
                pickle.dump(self.pixel_TotalVal, pix_Total)
    
    def check_create_data(self, recalculate):
        
        try:
            all_images_and_mask_paths = open(f"{self.files_path}hash/all_images_and_mask_paths_{self.out_cl_hash}.pickle", 'rb')
            self.all_images_and_mask_paths = pickle.load(all_images_and_mask_paths)
            
            if recalculate != False:
                raise ValueError('stuff is not in content')
        except:
            "Данные не сгруппированы"
            all_images_and_mask_paths = self.configurated_data()
            self.all_images_and_mask_paths = all_images_and_mask_paths
            with open(f"{self.files_path}hash/all_images_and_mask_paths_{self.out_cl_hash}.pickle", 'wb') as all_images_and_mask_paths:
                pickle.dump(self.all_images_and_mask_paths, all_images_and_mask_paths)
    
    def configurated_data(self):
        all_images_and_mask_paths = []
        masks_paths = self.masks_paths.copy()
        for x in self.images_paths:
            images_and_mask_paths = {}
            images_and_mask_paths["image"] = x
            images_and_mask_paths["mask"] = None        
            new_masks_paths = []
            for y in masks_paths:
                imag_name = x.split("/")[-1]
                mask_name = y.split("/")[-1]
                imag_name = imag_name.split(".")[:-1]
                mask_name = mask_name.split(".")[:-1]
                imag_name = ".".join(imag_name)
                mask_name = ".".join(mask_name)

                if imag_name==mask_name:
                    images_and_mask_paths["mask"] = y
                else:
                    new_masks_paths.append(y)

            masks_paths = new_masks_paths
            all_images_and_mask_paths.append(images_and_mask_paths)

        return all_images_and_mask_paths
            
    
    def to_out_classes(self, mask):
        
        size = mask.size()
        new_mask = torch.zeros(len(self.out_classes)+1,size[1],size[2])
        
        new_mask[0]=1
        
        if mask.max()>0:
            for out_class in self.out_classes:
                for i in out_class["summable_masks"]:
                    new_mask[out_class["id"],:,:][mask[i,:,:]==1]=1
                for i in out_class["subtractive_masks"]:
                    new_mask[out_class["id"],:,:][mask[i,:,:]==1]=0
                new_mask[0][new_mask[out_class["id"],:,:]==1]=0  
        
        return new_mask
    
    def load_data(self, path_to_file):
        if path_to_file==None:
            data = np.zeros((len(self.base_classes), self.resize[0], self.resize[1]))    
        else:
            if path_to_file.find("npz") !=-1:
                data = np.load(path_to_file)
            elif path_to_file.find("jpg")!=-1 or path_to_file.find("png")!=-1:
                data = cv2.imread(path_to_file)
                data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        return data
        
        
    def __getitem__(self, idx, for_inner_use = False):
                                              
        image_path = self.all_images_and_mask_paths[idx]["image"]
        mask_path = self.all_images_and_mask_paths[idx]["mask"]
        
        image = self.load_data(image_path)
        image = torch.tensor(image)
        
        base_mask = self.load_data(mask_path)
        base_mask = torch.tensor(base_mask)
                                              
        image = torch.unsqueeze(image, 0)
        image = (image-image.min())/(image.max()-image.min()+0.00000001)
                                      
        if base_mask.dim() == 2:
                base_mask = torch.unsqueeze(base_mask, 0)
        
        mask = torch.zeros(len(self.base_classes)+1, base_mask.size()[-2],  base_mask.size()[-1])
        mask[1:, :, :] = base_mask                           
        mask = self.to_out_classes(mask)
        
        if self.resize:                      
            image = torchvision.transforms.functional.resize(image, (self.resize))
            mask = torchvision.transforms.functional.resize(mask, (self.resize))
             
        if self.dataloader==False and for_inner_use == False:
            image = torch.unsqueeze(image, 0)
            mask = torch.unsqueeze(mask, 0)  
       
        result = {}
        result["images"] = image
        result["masks"] = mask
        result["labels"] = torch.amax(mask, dim=(-1, -2))
        result["values"] = torch.sum(mask, (-1,-2))
        return result


# это старые для желудка ниже для кишки
# Ainur_base_classes = [
    
#                     {'id': 1, 'name': 'SE', "summable_masks":[1], "subtractive_masks":[]},
#                     {'id': 2, 'name': 'GT', "summable_masks":[2], "subtractive_masks":[]},
#                     {'id': 3, 'name': 'NG', "summable_masks":[3], "subtractive_masks":[]},
#                     {'id': 4, 'name': 'F', "summable_masks":[4], "subtractive_masks":[]},
#                     {'id': 5, 'name': 'IM', "summable_masks":[5], "subtractive_masks":[]},
                    
#                     {'id': 6, 'name': 'LT', "summable_masks":[6], "subtractive_masks":[]},
#                     {'id': 7, 'name': 'GINL', "summable_masks":[7], "subtractive_masks":[]},
#                     {'id': 8, 'name': 'GINH', "summable_masks":[8], "subtractive_masks":[]},
#                     {'id': 9, 'name': 'TACG1', "summable_masks":[9], "subtractive_masks":[]},
#                     {'id': 10, 'name': 'TACG2', "summable_masks":[10], "subtractive_masks":[]},
                    
#                     {'id': 11, 'name': 'TACG3', "summable_masks":[11], "subtractive_masks":[]},
#                     {'id': 12, 'name': 'PACG1', "summable_masks":[12], "subtractive_masks":[]},
#                     {'id': 13, 'name': 'PACG2', "summable_masks":[13], "subtractive_masks":[]},
#                     {'id': 14, 'name': 'MPAC', "summable_masks":[14], "subtractive_masks":[]},
#                     {'id': 15, 'name': 'PCC', "summable_masks":[15], "subtractive_masks":[]},
                    
#                     {'id': 16, 'name': 'PCC-NOS', "summable_masks":[16], "subtractive_masks":[]},
#                     {'id': 17, 'name': 'MAC1', "summable_masks":[17], "subtractive_masks":[]},
#                     {'id': 18, 'name': 'MAC2', "summable_masks":[18], "subtractive_masks":[]},
#                     {'id': 19, 'name': 'ACLS', "summable_masks":[19], "subtractive_masks":[]},
#                     {'id': 20, 'name': 'HAC', "summable_masks":[20], "subtractive_masks":[]},
                    
#                     {'id': 21, 'name': 'ACFG', "summable_masks":[21], "subtractive_masks":[]},
#                     {'id': 22, 'name': 'SCC', "summable_masks":[22], "subtractive_masks":[]},
#                     {'id': 23, 'name': 'NDC', "summable_masks":[23], "subtractive_masks":[]},
#                     {'id': 24, 'name': 'NED', "summable_masks":[24], "subtractive_masks":[]}
                    
#                     ]


# это новые для кишки 
Ainur_base_classes = [
                    # {'id': 0, 'name': '0', "summable_masks":[0], "subtractive_masks":[]}, не згаю надо или нет
                    {'id': 1, 'name': 'GT', "summable_masks":[1], "subtractive_masks":[]},
                    {'id': 2, 'name': 'NGC', "summable_masks":[2], "subtractive_masks":[]},
                    {'id': 3, 'name': 'F', "summable_masks":[3], "subtractive_masks":[]},
                    {'id': 4, 'name': 'LT', "summable_masks":[4], "subtractive_masks":[]},
                    {'id': 5, 'name': 'SDL', "summable_masks":[5], "subtractive_masks":[]},
                    
                    {'id': 6, 'name': 'SDH', "summable_masks":[6], "subtractive_masks":[]},
                    {'id': 7, 'name': 'HPM', "summable_masks":[7], "subtractive_masks":[]},
                    {'id': 8, 'name': 'HPG', "summable_masks":[8], "subtractive_masks":[]},
                    {'id': 9, 'name': 'APL', "summable_masks":[9], "subtractive_masks":[]},
                    {'id': 10, 'name': 'APH', "summable_masks":[10], "subtractive_masks":[]},
                    
                    {'id': 11, 'name': 'TA', "summable_masks":[11], "subtractive_masks":[]},
                    {'id': 12, 'name': 'VA', "summable_masks":[12], "subtractive_masks":[]},
                    {'id': 13, 'name': 'INL', "summable_masks":[13], "subtractive_masks":[]},
                    {'id': 14, 'name': 'INH', "summable_masks":[14], "subtractive_masks":[]},
                    {'id': 15, 'name': 'ADCG1', "summable_masks":[15], "subtractive_masks":[]},
                    
                    {'id': 16, 'name': 'ADCG2', "summable_masks":[16], "subtractive_masks":[]},
                    {'id': 17, 'name': 'ADCG3', "summable_masks":[17], "subtractive_masks":[]},
                    {'id': 18, 'name': 'MAC', "summable_masks":[18], "subtractive_masks":[]},
                    {'id': 19, 'name': 'SRC', "summable_masks":[19], "subtractive_masks":[]},
                    {'id': 20, 'name': 'MC', "summable_masks":[20], "subtractive_masks":[]},
                    
                    {'id': 21, 'name': 'NDC', "summable_masks":[21], "subtractive_masks":[]},
                    {'id': 22, 'name': 'NED', "summable_masks":[22], "subtractive_masks":[]}          
                    ]


Ainur_out_classes = Ainur_base_classes

pochki_base_classes = [
                    # {'id': 0, 'name': '0', "summable_masks":[0], "subtractive_masks":[]}, не згаю надо или нет
                    {'id': 1, 'name': 'right_kidney_upper_segment_ID2', "summable_masks":[1], "subtractive_masks":[]},
                    {'id': 2, 'name': 'left_kidney_upper_segment_ID6', "summable_masks":[2], "subtractive_masks":[]},
                    {'id': 3, 'name': 'left_kidney_middle_segment_ID7', "summable_masks":[3], "subtractive_masks":[]},
                    {'id': 4, 'name': 'left_kidney_lower_segment_ID8', "summable_masks":[4], "subtractive_masks":[]},
                    {'id': 5, 'name': 'benign_tumor_ID10', "summable_masks":[5], "subtractive_masks":[]},
                    {'id': 6, 'name': 'abscess_ID12', "summable_masks":[6], "subtractive_masks":[]},
                    {'id': 7, 'name': 'cyst_ID11', "summable_masks":[7], "subtractive_masks":[]},
]


def get_direct_subdirectories(directory):
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return [os.path.join(directory, subdir) for subdir in subdirectories]

def get_number(file):
    print("file", file) # 
    parts = file.split('/')
    number = parts[6]
    second_chislo = parts[7]
    return number, second_chislo

def get_all_files(directory):
    all_files = []

    for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)

    return all_files

 

   
def save_to_papki(path, number_papki):
    
    # print("path", path)
    # print("number_papki", number_papki) # 78

    
    sct_coco = Universal_json_Segmentation_Dataset(json_file_path = path + "/",   
                                                        delete_list= [],
                                                        base_classes=pochki_base_classes,
                                                        out_classes=pochki_base_classes,
                                                        delete_null=False, # Fasle всегда 
                                                        resize= (256, 256),
                                                        dataloader = True,
                                                        recalculate = True # оставить True
                                                        )       
    
                
    colors = [((251, 206, 177), 'Абрикосовым'), ((127, 255,  212), 'Аквамариновым'), ((255, 36, 0), 'Алым'), ((153, 102, 204), 'Аметистовым'),
                        ((153, 0, 102), 'Баклажановым'), ((48, 213, 200), 'Бирюзовым'), ((152, 251,  152), 'Бледно зеленым'),
                        ((213, 113, 63), 'Ванильным'), ((100, 149,  237), 'Васильковым'), ((34,139,34), 'Зелёный лесной'), ((0,0,255), 'Синий'),
                        ((75,0,130), 'Индиго'), ((255,0,255), 'Чёрный'), ((0,51,  153), 'Маджента'), ((65,105,225), 'Королевский синий'),
                        ((255,255,0), 'Жёлтый'), ((255,69,0), 'Оранжево-красный'), ((255,0,0), 'Темно синим'), ((0,51,  153), 'Красный'),
                        ((255,215,0), 'Золотой'), ((250,128,114), 'Лососевый'), ((255,99,71), 'Томатный'), ((255,215,0), 'Золотой'),
                        ((0,139,139), 'Тёмный циан'), ((0,255,255), 'Морская волна')]
            
    # это старые для желудка
    # list_of_name_out_classes = ["SE", "GT", "NG", "F", "IM", "LT", "GINL", "GINH", "TACG1", "TACG2", "TACG3", "PACG1", "PACG2", 
    #                             "MPAC", "PCC", "PCC-NOS", "MAC1", "MAC2", "ACLS", "HAC", "ACFG", "SCC", "NDC", "NED"]

    # а это новые для кишки
    list_of_name_out_classes = [ "GT", "NGC", "F", "LT", "SDL", 
                                "SDH", "HPM", "HPG", "APL", "APH", 
                                "TA", "VA", "INL", "INH", "ADCG1", 
                                "ADCG2", "ADCG3", "MAC", "SRC", "MC", 
                                "NDC", "NED", ]

    # print("sct_coco.all_img_list", sct_coco.all_img_list)

    for k in sct_coco.all_img_list:  
        # print("k", k)               
        result = sct_coco[k] 
        image = result["images"]
        label = result["labels"]
        label = label[1:]
        # print("label", label)
        # print("label shape", label.shape)
        if sum(label) == 1:
            if label.max().item() != 0:
                clas = label.argmax().item() + 1
                # тут сделать 
                # classes = 
            # print(label.shape)
            # print(label[1:], label[1:].amax(), label[1:].argmax())
                mask = result["masks"]
                rgb_image = result["rgb_image"]
                print("rgb_image shape", rgb_image.shape) # (256, 256, 3)
                print("mask shape", mask.shape) # torch.Size([25, 256, 256])
                
                mask = mask.detach().numpy()
                
                for i in range(np.shape(mask)[0]):
                        
                        # print("mask[i] shape", mask[i].shape) # (256, 256)
                        contours, h = cv2.findContours(mask[i].astype(int).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        # print("contours", contours)
                        rgb_image = cv2.drawContours(rgb_image, contours, -1, (colors[i][0][0], colors[i][0][1], colors[i][0][2]), 2)
                        if np.max(mask[i])==1 and i!=0:
                            
                        
                            
                            text = list_of_name_out_classes[i]+ " " +str(np.max(mask[i]))
                            # plt.text(2000, k, text , color = (self.colors[i][0][0]/255, self.colors[i][0][1]/255, self.colors[i][0][2]/255)) 
                            cv2.putText(rgb_image, f"label class: {text}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA) 
                            # k+=50
                
                
                ####################
                # отрисовал маску отдельно
                # mask_path = f"/home/imran-nasyrov/sct_project/sct_data/just_mask_plt/{list_of_name_out_classes[clas]}"
                # # print("clas", clas)
                # # print("mask shape", mask.shape)
                # # print("mask[clas, :, :]", mask[clas, :, :])
                # if not os.path.exists(mask_path):
                #     os.makedirs(mask_path)
                    
                # my_result = cv2.normalize(mask[clas, :, :], dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                # my_pixel_array = cv2.cvtColor(my_result, cv2.COLOR_RGB2BGR)
                
                # cv2.imwrite(f"{mask_path}/output_image_{number_papki}.jpg", my_pixel_array)
                ####################
                    
                            
                            
                papki_path = f"/home/imran-nasyrov/output_guts2/{list_of_name_out_classes[clas]}"
                
                if not os.path.exists(papki_path):
                    os.makedirs(papki_path)
                
                photo_path = f"{papki_path}/output_image_{number_papki}_{k}.jpg" 
                # print("photo_path", photo_path, k)
                # print("k", k)
                # print("photo_path", photo_path)
                
                all_files = get_all_files("/home/imran-nasyrov/output_guts2/" + str(list_of_name_out_classes[clas]))
                # print("all_files", all_files)
                if len(all_files) < 50: # убрать это
                    cv2.imwrite(photo_path, rgb_image)
                    print(f"file saved to {photo_path}")
        
        
#########################################################################################

# # path = "/home/imran-nasyrov/sct_project/sct_data/CONVERT_AINUR/"
# path = "/mnt/netstorage/Medicine/Medical/stomach_json_part_2"
# subdirectories_list = get_direct_subdirectories(path)
    
# # print("subdirectories_list", subdirectories_list)
# with tqdm(total=len(subdirectories_list), desc="Making files") as pbar_dirs:      
#     for i in subdirectories_list:
#         # try:
#         # print("i", i) # i /home/imran-nasyrov/sct_project/sct_data/CONVERT_AINUR/78
#         # all_files = get_all_files(i)
        
#         parts = i.split('/')
#         id_number = parts[-1]
#         print("id_number", id_number) # id_number 78

        
#         save_to_papki(i, id_number) 
#         # except:
#         #     print("no")
        
#         pbar_dirs.update(1)  
  
# print("ok")

#########################################################################################


from coco_classes import sinusite_base_classes, sinusite_pat_classes_3


def save_to_papki_OD(path, number_papki):
    sct_coco = Universal_json_Segmentation_Dataset(json_file_path=path + "/",   
                                                    delete_list=[], 
                                                    base_classes=sinusite_base_classes, 
                                                    out_classes=sinusite_pat_classes_3, 
                                                    delete_null=False,  # Fasle всегда 
                                                    resize=(256, 256), 
                                                    dataloader=True, 
                                                    recalculate=False  # оставить True
                                                    )       

    colors = [((251, 206, 177), 'Абрикосовым'), ((127, 255,  212), 'Аквамариновым'), ((255, 36, 0), 'Алым'), ((153, 102, 204), 'Аметистовым'),
              ((153, 0, 102), 'Баклажановым'), ((48, 213, 200), 'Бирюзовым'), ((152, 251,  152), 'Бледно зеленым'),
              ((213, 113, 63), 'Ванильным'), ((100, 149,  237), 'Васильковым'), ((34,139,34), 'Зелёный лесной'), ((0,0,255), 'Синий'),
              ((75,0,130), 'Индиго'), ((255,0,255), 'Чёрный'), ((0,51,  153), 'Маджента'), ((65,105,225), 'Королевский синий'),
              ((255,255,0), 'Жёлтый'), ((255,69,0), 'Оранжево-красный'), ((255,0,0), 'Темно синим'), ((0,51,  153), 'Красный'),
              ((255,215,0), 'Золотой'), ((250,128,114), 'Лососевый'), ((255,99,71), 'Томатный'), ((255,215,0), 'Золотой'),
              ((0,139,139), 'Тёмный циан'), ((0,255,255), 'Морская волна')]

    list_of_name_out_classes = ["GT", "NGC", "F", "LT", "SDL", 
                                "SDH", "HPM", "HPG", "APL", "APH", 
                                "TA", "VA", "INL", "INH", "ADCG1", 
                                "ADCG2", "ADCG3", "MAC", "SRC", "MC", 
                                "NDC", "NED"]

    for k in sct_coco.all_img_list:                  
        result = sct_coco[k] 
        image = result["images"]
        label = result["labels"]
        label = label[1:]
        if sum(label) == 1:
            if label.max().item() != 0:
                clas = label.argmax().item() + 1
                mask = result["masks"]
                rgb_image = image#result["rgb_image"]
                # print("rgb_image shape", rgb_image.shape)
                # print("image shape", image.shape)
                
                # if isinstance(rgb_image, torch.Tensor):
                rgb_image = rgb_image.squeeze().detach().numpy() #.permute(1, 2, 0)
                rgb_image = (rgb_image * 255).astype(np.uint8)

                # rgb_image = image.detach().numpy()
                # rgb_image = cv2.cvtColor(image[0], cv2.COLOR_GRAY2RGB)
                # rgb_image = torch.squeeze(rgb_image*255)#.detach().numpy()
                # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                
                
                mask = mask.detach().numpy()

                # for i in range(np.shape(mask)[0]):
                #     color = colors[i % len(colors)][0]
                    
                #     contours, _ = cv2.findContours(mask[i].astype(int).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                #     rgb_image = cv2.drawContours(rgb_image, contours, -1, (color[0], color[1], color[2]), 2)
                    
                #     if np.max(mask[i]) == 1 and i != 0:
                #         x, y, w, h = cv2.boundingRect(mask[i].astype(int).astype(np.uint8))
                #         cv2.rectangle(rgb_image, (x, y), (x + w, y + h), color, 2)
                        
                #         text = list_of_name_out_classes[i]
                #         cv2.putText(rgb_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                for i in range(np.shape(mask)[0]):
                    contours, _ = cv2.findContours(mask[i].astype(int).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    rgb_image = cv2.drawContours(rgb_image, contours, -1, (colors[i][0][0], colors[i][0][1], colors[i][0][2]), 2)

                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        color = colors[i % len(colors)][0]
                        cv2.rectangle(rgb_image, (x, y), (x + w, y + h), color, 2)

                    if np.max(mask[i]) == 1 and i != 0:
                        text = list_of_name_out_classes[i] + " " + str(np.max(mask[i]))
                        cv2.putText(rgb_image, f"label class: {text}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

                papki_path = f"/home/imran-nasyrov/output_guts2/{list_of_name_out_classes[clas]}"
                
                if not os.path.exists(papki_path):
                    os.makedirs(papki_path)
                
                photo_path = f"{papki_path}/output_image_{number_papki}_{k}.jpg"
                
                all_files = get_all_files("/home/imran-nasyrov/output_guts2/" + str(list_of_name_out_classes[clas]))
                if len(all_files) < 10:
                    cv2.imwrite(photo_path, rgb_image)



# Это чисто чтобы одну папку проверить 
# path = "/home/imran-nasyrov/111303"
# path = "/mnt/netstorage/Medicine/Medical/guts_json_01_07/126"
# path = "/home/imran-nasyrov/sinusite_json_data/task_sinusite_data_29_11_23_1_st_sin_labeling"
path = "/home/imran-nasyrov/Zaytseva TI_2 Non Contrast  3.0  I41s  4"

parts = path.split('/')
id_number = parts[-1]
print("id_number", id_number) # id_number 78


save_to_papki(path, id_number) 
# save_to_papki_OD(path, id_number)