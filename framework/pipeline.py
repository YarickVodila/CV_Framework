from framework.moduls.detection import YoloDetection
from framework.moduls.classification import EfficientNetClassification
from framework.moduls.segmentation import YoloSegmentation
from framework.moduls.face_analyze import FaceAnalyze
from framework.moduls.face_recognition import FaceRecognition
from framework.moduls.action_classification import ActionClassification

from ultralytics import YOLO
from tensorflow import keras
import torch
import os

import numpy as np

from typing import Any, Dict, List, Tuple, Union, Optional

import json

class Pipeline:
    def __init__(self):
        """
        Конструктор класса Pipeline.

        """
        
        self.base_comp = ['classification', 'detection', 'segmentation']
        self.face_comp = ['face_analyze', 'face_recognition']
        self.action_cls_comp = ['action_classification']

        self.pipeline_objs = {}
        self.pipeline_kwargs = {}

    def __name__(self):
        return 'pipeline'
    
    def add_pipe(self, name, **kwargs):
        
        if name == 'classification':
            self.pipeline_objs['classification'] = EfficientNetClassification(**kwargs)
            self.pipeline_kwargs['classification'] = kwargs

        elif name == 'detection':
            self.pipeline_objs['detection'] = YoloDetection(**kwargs)
            self.pipeline_kwargs['detection'] = kwargs

        elif name == 'segmentation':
            self.pipeline_objs['segmentation'] = YoloSegmentation(**kwargs)
            self.pipeline_kwargs['segmentation'] = kwargs

        elif name == 'face_analyze':
            self.pipeline_objs['face_analyze'] = FaceAnalyze(**kwargs)
            self.pipeline_kwargs['face_analyze'] = kwargs

        elif name == 'face_recognition':
            self.pipeline_objs['face_recognition'] = FaceRecognition(**kwargs)
            self.pipeline_kwargs['face_recognition'] = kwargs
        
        elif name == 'action_classification':
            self.pipeline_objs['action_classification'] = ActionClassification(**kwargs)
            self.pipeline_kwargs['action_classification'] = kwargs

        else:
            raise ValueError("""
                Такого компонента не существует.
                Доступные компоненты Pipeline: 
                             'classification', 
                             'detection', 
                             'segmentation', 
                             'face_analyze', 
                             'face_recognition', 
                             'action_classification'
                """)
        

        if any([True if copm in self.base_comp else False for copm in self.pipeline_objs.keys()]) and \
            any([True if copm in self.face_comp else False for copm in self.pipeline_objs.keys()]):

            # Удаляем добавленный компонент
            
            self.del_pipe(name)
            
            raise ValueError(
                """
                Все компоненты Pipeline должны быть для одной из задач: 
                1) 'classification', 'detection', 'segmentation' - для задач связанных работой любых изображений
                2) 'face_analyze', 'face_recognition' - для задач связанных с лицами
                3) 'action_classification' - для задач связанных с классификацией действий
                """
            )
        elif any([True if copm in self.base_comp else False for copm in self.pipeline_objs.keys()]) and \
            any([True if copm in self.action_cls_comp else False for copm in self.pipeline_objs.keys()]):

            # Удаляем добавленный компонент
            
            self.del_pipe(name)
            
            raise ValueError(
                """
                Все компоненты Pipeline должны быть для одной из задач: 
                1) 'classification', 'detection', 'segmentation' - для задач связанных работой любых изображений
                2) 'face_analyze', 'face_recognition' - для задач связанных с лицами
                3) 'action_classification' - для задач связанных с классификацией действий
                """
            )
        
        elif any([True if copm in self.face_comp else False for copm in self.pipeline_objs.keys()]) and \
            any([True if copm in self.action_cls_comp else False for copm in self.pipeline_objs.keys()]):

            # Удаляем добавленный компонент

            self.del_pipe(name)

            raise ValueError(
                """
                Все компоненты Pipeline должны быть для одной из задач: 
                1) 'classification', 'detection', 'segmentation' - для задач связанных работой любых изображений
                2) 'face_analyze', 'face_recognition' - для задач связанных с лицами
                3) 'action_classification' - для задач связанных с классификацией действий
                """
            )
        

    def del_pipe(self, name):
        del_copmonent = self.pipeline_objs.pop(name, False)
        del_keywards = self.pipeline_kwargs.pop(name, None)

        torch.cuda.empty_cache()

        if del_copmonent == False:
            raise ValueError("Такого компонента не существует в вашем Pipeline")
    

    def predict(self, image: Union[str, np.ndarray, List], **kwargs) -> Dict:

        is_base = any([True if copm in self.base_comp else False for copm in self.pipeline_objs.keys()])
        is_face = any([True if copm in self.face_comp else False for copm in self.pipeline_objs.keys()])
        is_action = any([True if copm in self.action_cls_comp else False for copm in self.pipeline_objs.keys()])

        # Результат работы каждого компонента Pipeline
        result = {key:None for key in self.pipeline_objs.keys()}

        if is_base:
            for key in self.pipeline_objs.keys():
                result[key] = self.pipeline_objs[key].predict(image)

        elif is_face:
            if isinstance(image, List):
                for key in self.pipeline_objs.keys():
                    result[key] = [self.pipeline_objs[key].predict(img, **kwargs) for img in image]
                    

            elif isinstance(image, np.ndarray):
                # Если массив изображений
                if len(image.shape) == 4:
                    for key in self.pipeline_objs.keys():
                        result[key] = [self.pipeline_objs[key].predict(img, **kwargs) for img in image]

                # Если одно изображение
                elif len(image.shape) == 3:
                    for key in self.pipeline_objs.keys():
                        result[key] = [self.pipeline_objs[key].predict(image, **kwargs)]
            elif isinstance(image, str):
                for key in self.pipeline_objs.keys():
                    result[key] = [self.pipeline_objs[key].predict(image, **kwargs)]

        elif is_action:
            for key in self.pipeline_objs.keys():
                result[key] = self.pipeline_objs[key].predict(image)

        return result
    

    #TODO: Сделать сохранение Pipeline
    def save_pipeline(self, path_name: str = 'pipeline'):
        # torch.save(self.pipeline_objs, path)

        try:
            # Создаём папку, если она не существует
            os.chdir(os.getcwd())
            os.makedirs(path_name)
        except:
            pass


        components = {comp:{'model_name': None, 'kwargs': None} for comp in self.pipeline_objs.keys()}

        for key in self.pipeline_objs.keys():
            
            components[key]["kwargs"] = self.pipeline_kwargs[key]

            if key == "classification":
                torch.save(self.pipeline_objs[key].model, os.path.join(path_name, 'model_classification.pth'))
                components[key]["model_name"] = "model_classification.pth"

            elif key == "detection":
                self.pipeline_objs[key].model.save(os.path.join(path_name, 'model_detection.pt'))
                components[key]["model_name"] = "model_detection.pt"

            elif key == "segmentation":
                self.pipeline_objs[key].model.save(os.path.join(path_name, 'model_segmentation.pt'))
                components[key]["model_name"] = "model_segmentation.pt"

            elif key == "action_classification":
                self.pipeline_objs[key].model.save(os.path.join(path_name, 'model_action_classification'))
                components[key]["model_name"] = "model_action_classification"

            elif key == "face_analyze":
                pass

            elif key == "face_recognition":
                pass
        
        with open(os.path.join(path_name, 'components.json'), 'w') as fp:
            json.dump(components, fp)

    def load_pipeline(self, path_name: str = 'pipeline'):

        with open(os.path.join(path_name, 'components.json'), 'r') as fp:
            # Загружаем компоненты Pipeline и их параметры
            components = json.load(fp)

        
        for comp in components:
            if comp == 'classification':
                self.pipeline_objs[comp] = EfficientNetClassification(**components[comp]['kwargs'])
                self.pipeline_objs[comp].model = torch.load(os.path.join(path_name, components[comp]['model_name']))
                self.pipeline_kwargs[comp] = components[comp]['kwargs']
                

            elif comp == 'detection':
                self.pipeline_objs[comp] = YoloDetection(**components[comp]['kwargs'])
                self.pipeline_objs[comp].model = YOLO(os.path.join(path_name, components[comp]['model_name']), task='detect')
                self.pipeline_kwargs[comp] = components[comp]['kwargs']

            elif comp == 'segmentation':
                self.pipeline_objs[comp] = YoloSegmentation(**components[comp]['kwargs'])
                self.pipeline_objs[comp].model = YOLO(os.path.join(path_name, components[comp]['model_name']), task='segment')
                self.pipeline_kwargs[comp] = components[comp]['kwargs']

            elif comp == 'action_classification':
                self.pipeline_objs[comp] = ActionClassification(**components[comp]['kwargs'])
                self.pipeline_objs[comp].model = keras.models.load_model(os.path.join(path_name, components[comp]['model_name']))
                self.pipeline_kwargs[comp] = components[comp]['kwargs']

            elif comp == 'face_analyze':
                self.pipeline_objs[comp] = FaceAnalyze(**components[comp]['kwargs'])
                self.pipeline_kwargs[comp] = components[comp]['kwargs']


            elif comp == 'face_recognition':
                self.pipeline_objs[comp] = FaceRecognition(**components[comp]['kwargs'])
                self.pipeline_kwargs[comp] = components[comp]['kwargs']
            


        # for key in components.keys():
