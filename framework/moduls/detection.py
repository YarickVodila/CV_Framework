from ultralytics import YOLO
import numpy as np
import torch
import os
import cv2

class YoloDetection:
    def __init__(self, model_size:str, yolo_result=True, **kwargs_for_predict):
        """
        Конструктор класса YoloDetection.

        Параметры:
            model_size (str): Размер модели YOLO. Допустимые значения: 'n', 's', 'm', 'l', 'x'.
            yolo_result (bool): Опция для возврата результата в формате YOLO ultralytics. По умолчанию True
            **kwargs_for_predict: Дополнительные аргументы для метода predict модели.

        Примеры использования:
            ```py
            detection = YoloDetection(model_size='m', verbose=False, conf=0.6)
            ```

        """
        assert model_size in ["n", "s", "m", "l", "x"], "Размер модели должен быть 'n', 's', 'm', 'l', 'x'"

        self.yolo_result = yolo_result

        # Загрузка модели
        self.model = YOLO(f"yolov8{model_size}.pt")

        self.kwargs_for_predict = kwargs_for_predict

    def __name__(self):
        return 'detection'

    def predict(self, images:str | list | tuple):
        """ 
        Предсказывает объекты на изображениях.

        Параметры:
            images (str | list | tuple): Путь к изображению или список путей к изображениям для предсказания.

        Возвращает:
            - Если yolo_result=True, возвращает список результатов YOLO (result[i].boxes подробнее смотреть по ссылке https://docs.ultralytics.com/ru/modes/predict/#boxes).
            - Если yolo_result=False, возвращает словарь с информацией о предсказанных объектах на каждом изображении. Пример ниже
                
                ```py
                {
                    'файл.png': {
                        "classes": np.array([0]),
                        "str_classes": ["person"],
                        "bboxs": np.array([[550.25, 690.15, 825.81, 1016.4]], dtype=float32),
                        "conf": np.array([0.7562], dtype=float32)
                    }
                }
                ```
        Примеры использования:
            ```py
            results = detection.predict(images=["image1.jpg", "image2.jpg"], yolo_result=True)
            results_custom = detection.predict("image3.jpg", yolo_result=False)
            ```

        """
        results = self.model.predict(source=images, **self.kwargs_for_predict)

        if self.yolo_result:
            return [res.boxes for res in results]
        
        else:
            custom_result = {}

            for res in results:
                
                # bbox в формате xywh numpy, где 
                # x, y - координаты центра изображения. 
                # w, h - ширина и высота ббокса 

                bbox = res.boxes.xywh.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy() # Номера классов
                str_classes = [self.model.names[cls] for cls in classes] # Расшифрованные классы
                conf = res.boxes.conf.cpu().numpy() # Порог уверенности в каждом классе

                custom_result[os.path.basename(res.path)] = {
                    "classes": classes,
                    "str_classes": str_classes,
                    "bboxs": bbox,
                    "conf": conf,
                }

            return custom_result
        