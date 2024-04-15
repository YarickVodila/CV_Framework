from typing import Any, Dict, List, Tuple, Union, Optional

from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

from torchmetrics import AUROC

from ultralytics import YOLO

import json
import os

import PIL

import cv2

from framework.pipeline import Pipeline

import pandas as pd
import numpy as np


class ExampleDataset(Dataset):
    def __init__(self, x, y):
        self.len = len(x)
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



class Trainer:
    def __init__(self, pipeline: object):
        """"""

        self.pipeline = pipeline

        face_comp = ['face_analyze', 'face_recognition']

        if any([True if copm in face_comp else False for copm in self.pipeline.pipeline_objs.keys()]):
            ValueError("Модели анализа лиц не подлежат тренировки!!")

        # Сохраняем pipeline
        # self.pipeline.save_pipeline()

    def train(self, dataset_path, data_x:np.array = None, data_y:np.array = None, epochs:int=10, batch_size:int=4, **kwargs):

        with open(os.path.join(dataset_path, 'config.json'), 'r') as fp:
            # Загружаем конфиг для тренировки
            config = json.load(fp)

        for comp in self.pipeline.pipeline_objs.keys():
            if comp == 'classification' and config.get(comp, False):
                self.pipeline.pipeline_objs["classification"].model = self.__train_classification(
                    self.pipeline.pipeline_objs["classification"].model, 
                    device = self.pipeline.pipeline_objs["classification"].device,  
                    **config[comp]
                )

            elif comp == 'detection' and config.get(comp, False):
                self.pipeline.pipeline_objs["detection"].model = self.__train_yolo_models(
                    self.pipeline.pipeline_objs["detection"].model,
                    **config[comp]
                )
            
            elif comp == 'segmentation' and config.get(comp, False):
                self.pipeline.pipeline_objs["segmentation"].model = self.__train_yolo_models(
                    self.pipeline.pipeline_objs["segmentation"].model,
                    **config[comp]
                )

            elif comp == "action_classification" and config.get(comp, False):
                self.pipeline.pipeline_objs["action_classification"].model = self.pipeline.pipeline_objs["action_classification"].model.fit(
                    data_x, 
                    data_y, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    **kwargs
                )
                



        return self.pipeline
            
    def __train_yolo_models(self,model, **kwargs):
        """ 
        Приватный метод для тренировки компонентов детекции и сегментации

        
        """

        results = model.train(**kwargs)

        return model



    def __train_classification(self, model, device, image_train_path:str, labels_train_path:str, image_val_path:str = None, labels_val_path:str = None, n_classes:int = 2,  epochs:int = 10, batch_size = 32, **kwargs):
        """ 
        Приватный метод для тренировки компонента классификации

        
        """
        
        is_val = False

        optimizer = optim.Adam(model.parameters(), lr=0.001)


        task = "multiclass" if n_classes > 2 else "binary"
        auroc = AUROC(task = task, num_classes=n_classes).to(device)


        annotation_train = pd.read_csv(labels_train_path)

        dataset_train, criterion = self.__prepare_dataset(image_train_path, annotation_train, n_classes, device)

        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

        # Если валидация есть
        if isinstance(image_val_path, str):
            annotation_val = pd.read_csv(labels_val_path)

            dataset_val, criterion = self.__prepare_dataset(image_val_path, annotation_val, n_classes, device)

            val_loader = DataLoader(dataset_val, batch_size=dataset_val.__len__(), shuffle=False)
            
            is_val = True

        all_losses_train = []
        all_metrics_train = []

        all_losses_val = []
        all_metrics_val = []
        
        for ep in tqdm(range(epochs), desc = "Эпоха"):
            print("\n", "_"*15, f"Эпоха: {ep}", "_"*15, "\n")

            batch_loss_train = 0
            batch_metrics_train = 0

            batch_loss_val = 0
            batch_metrics_val = 0
            
            for features, target in train_loader:
                optimizer.zero_grad()
                output = model(features)
                loss = criterion(output, target)

                batch_loss_train += float(loss.float())
                print("Loss-train: ", float(loss.float()))
                batch_metrics_train += float(auroc(output, target).cpu()) 

                loss.backward()
                optimizer.step()
                
            
            all_losses_train.append(batch_loss_train)
            all_metrics_train.append(batch_metrics_train)

            print('\n\n')

            if is_val:
                for features, target in val_loader:
                    optimizer.zero_grad()
                    with torch.no_grad():
                        output = model(features)
                        loss = criterion(output, target)
                        batch_loss_val += float(loss.float())
                        print("Loss-val: ", float(loss.float()))
                        
                        batch_metrics_val += float(auroc(output, target).cpu())   
                        
                all_losses_val.append(batch_loss_val)
                all_metrics_val.append(batch_metrics_val)
            
            print('\n')
            print('_' * 50)
        return model
        


    def __prepare_dataset(self, image_path, annotation, n_classes = 2, device = 'cpu'):
        """ 
        Приватный метод для подготовки датасета

        """
        
        # Превращаем id в путь до изображений
        annotation["id"] = annotation["id"].apply(lambda x: os.path.join(image_path, x))

        # Загружаем изображения и превращаем в тензор
        image = torch.tensor([self.__resize_img(np.array(PIL.Image.open(path).convert('RGB'))) for path in annotation["id"]], dtype = torch.float32).to(device)
        labels = annotation["target"].to_numpy() # torch.tensor(annotation["label"], dtype = torch.int64)
        
        # Бинарная классификация
        if n_classes == 2:
            criterion = nn.BCELoss()
            labels = torch.tensor(labels.reshape(-1, 1), dtype = torch.float32)

            dataset = ExampleDataset(image.to(device), labels.to(device))

        # Многоклассовая классификация
        else:
            criterion = nn.CrossEntropyLoss()
            ohe_labels = torch.zeros(image.shape[0], n_classes)        
            ohe_labels[torch.arange(image.shape[0]), labels] = 1.0

            dataset = ExampleDataset(image.to(device), ohe_labels.to(device))

        
        return dataset, criterion

            


    def __resize_img(self, img:np.array) -> np.array: 
        """
        "Ресайз" изображения до заданных размеров (224, 224).
        
        Args:
            img: np.array Исходное изображение
        
        Returns:
            torch.tensor "Ресайзнутое" изображение в формате torch.tensor
        """

        # Задание новых размеров изображения
        new_width, new_height = 224, 224 
        
        # Определение текущих размеров изображения
        height, width = img.shape[:2]
        
        if height>new_height or width>new_width: # Если вдруг изображение больше нашего нового размера
            max_size = max(height, width)
            scale_proc = new_width / max_size
            dim = (int(width * scale_proc), int(height * scale_proc))
            
            # print('До ресайза: ', img.shape)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            # print('После ресайза: ', img.shape)
            
            
        # Определение текущих размеров изображения
        height, width = img.shape[:2]

        # Создание пустого изображения с черным фоном
        resized_img = np.zeros((new_height, new_width, 3), np.uint8)
        #resized_img[:, :] = (0, 0, 0)

        # Вычисление координат для вставки исходного изображения посередине
        x_offset = int((new_width - width) // 2)
        y_offset = int((new_height - height) // 2)

        # Если надо вставить в левый верхний угол ставим 0 0. Пример ниже 
        # x_offset = 0
        # y_offset = 0

        # Вставка исходного изображения в новое изображение
        resized_img[y_offset:y_offset+height, x_offset:x_offset+width] = img

        # Изменение порядка каналов Height, Wight, Channel -> Channel, Height, Wight
        # resized_img = torch.tensor(np.array(resized_img.transpose(2, 0, 1), dtype=np.float32))
        resized_img = np.array(resized_img.transpose(2, 0, 1), dtype=np.float32)

        return resized_img/255

