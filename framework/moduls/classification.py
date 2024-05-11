import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l

import warnings

import numpy as np

import cv2

from typing import Any, Dict, List, Tuple, Union, Optional


""" 
Изменение порядка каналов в изображении (Weight Height Chanel) -> (Chanel, Weight Height)

state_val = torch.tensor(np.array([state_val.transpose(2, 0, 1)], dtype=np.float32))
"""

class EfficientNetClassification:
    def __init__(self, model_size:str = "s", transfer_learning:bool = True, layers: list | tuple = None, device: str = 'cuda'):
        """
        Initialize the EfficientNetClassification model with the specified parameters.

        Parameters:
            model_size (str): The size of the model ('s', 'm', or 'l').
            transfer_learning (bool): Flag indicating if transfer learning should be used.
            layers (list | tuple): The layers to be added to the classification model.
            device (str): The device to load the model on ('cuda' or 'cpu').

        Returns:
            None
            
        """

        
        if torch.device("cuda" if torch.cuda.is_available() else "cpu") == "cpu":
            self.device = 'cpu'

        elif torch.device("cuda" if torch.cuda.is_available() else "cpu") == "cuda":
            self.device = device
        
        else:
            self.device = device

        """
        layers: list | tuple: порядок слоев в модели для классификации
        Пример:
            (nn.Dropout(p=0.2, inplace=True), nn.Linear(in_features=1280, out_features=2, bias=True))
        """

        assert model_size in ["s", "m", "l"], "Размер модели должен быть 's', 'm' или 'l'"

        # Загрузка модели
        if model_size == "s":
            self.model = efficientnet_v2_s(weights= "IMAGENET1K_V1")
        elif model_size == "m":
            self.model = efficientnet_v2_m(weights= "IMAGENET1K_V1")
        elif model_size == "l":
            self.model = efficientnet_v2_l(weights= "IMAGENET1K_V1")


        # Если хотим использовать трансферное обучение
        if transfer_learning:
            # Заморозка весов
            for param in self.model.parameters():
                param.requires_grad = False

            # Разморозка весов классификации
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        else:
            print("Используется стандартная модель, с обучением всех слоёв!!!")


        # Добавляем новые слои классификации 
        if layers:
            self.model.classifier = nn.Sequential()
            for layer in layers:
                self.model.classifier.append(layer)
        
        # Если хотим использовать стандартные слои предложенные авторам модели с двумя нейронами на выходе
        # elif not layers and is_trainable:
        #     self.model.classifier = nn.Sequential(
        #         nn.Dropout(p=0.2, inplace=True),
        #         nn.Linear(in_features=1280, out_features=2, bias=True),
        #     )
        
        # Если не хотим ничего менять в модели

        else:
            print("Модель имеет на выходе 1000 классов, как в датасете ImageNet1K")
            print("Подробная информация о датасете доступна по ссылке  https://paperswithcode.com/dataset/imagenet-1k-1")


        self.model.to(self.device)
        
    def __name__(self):
        return 'classification'

    def resize_img(self, img:np.array) -> torch.tensor: 
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
       

    def predict(self, images: Union[str, np.ndarray, List], return_probs: bool = False) -> torch.tensor:
        # Перевод массива изображений в тензор
        # print(np.array([self.resize_img(img) for img in images]).shape)

        if isinstance(images, str):
            images = cv2.imread(images, cv2.COLOR_BGR2RGB)
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
            tensor_images = torch.tensor([self.resize_img(images)], dtype=torch.float32)

        elif isinstance(images, List):
            if isinstance(images[0], str):
                images = [cv2.imread(image, cv2.COLOR_BGR2RGB) for image in images]
                images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
                tensor_images = torch.tensor(np.array([self.resize_img(img) for img in images]), dtype=torch.float32)
        
        else:
            if len(images.shape) == 3: # Если подаётся одно изображение
                tensor_images = torch.tensor(np.array([self.resize_img(images)]), dtype=torch.float32)
            else:
                tensor_images = torch.tensor(np.array([self.resize_img(img) for img in images]), dtype=torch.float32)
        
        # print(tensor_images.shape)
        if return_probs:
            result = self.model(tensor_images)
        else:
            result = torch.argmax(self.model(tensor_images), dim=1)

        return result
