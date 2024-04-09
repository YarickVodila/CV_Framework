from deepface import DeepFace
import os
import numpy as np

from typing import Any, Dict, List, Tuple, Union, Optional
import json

class FaceAnalyze:
    def __init__(self, save_results:bool = False, save_path:str = ""):
        """
            
        Args:
            save_results (bool, optional): Необходимо ли сохранять результат анализа в файл `"analyze_id.json"`. По умолчанию False.
            save_path (str, optional): Путь до директории, в которую необходимо сохранять результат анализа. По умолчанию "".
        """
        self.save_results = save_results
        self.save_path = save_path

    def __name__(self):
        return 'face_analyze'

    def predict(self, image: Union[str, np.ndarray], id:int = 0, **kwargs):
        """
        Args:
            image (Union[str, np.ndarray]): Путь до исходного изображения или открытое изображение 
            id (int, optional): Идентификатор изображения. По умолчанию 0.

        Returns:

            List: Список с результатами анализа. 

            - 'region' (dict): Координаты текущей области лица.
            - 'x': x-координата верхнего левого угла лица.
            - 'y': y-координата верхнего левого угла лица.
            - 'w': ширина обнаруженной области лица.
            - 'h': Высота области обнаруженного лица.

        - 'age' (float): Приблизительный возраст обнаруженного лица.

        - 'face_confidence' (float): показатель достоверности для обнаруженного лица.
            Указывает на надежность распознавания лица.

        - 'dominant_gender' (str): Доминирующий пол обнаруженного лица.
            Либо "Мужчина", либо "Женщина".

        - 'gender' (dict): Показатель достоверности для каждой гендерной категории.
            - 'Man': Confidence score для мужского пола.
            - 'Woman': Confidence score для женского пола.

        - 'dominant_emotion' (str): доминирующая эмоция на обнаруженном лице.
            Возможные значения "sad," "angry," "surprise," "fear," "happy,"
            "disgust," and "neutral"

        - 'emotion' (dict): Confidence scores для каждой категории эмоций.
            - 'sad': оценка уверенности для грусти.
            - 'angry': оценка уверенности для гнева.
            - 'surprise': оценка уверенности за удивление.
            - 'fear': оценка уверенности страха.
            - 'happy': оценка уверенности счастье.
            - 'disgust': оценка уверенности отвращение.
            - 'neutral': оценка уверенности нейтральности.

        - 'dominant_race' (str): доминирующая раса у обнаруженного лица.
            Возможные значения "indian," "asian," "latino hispanic,"
            "black," "middle eastern," and "white."

        - 'race' (dict): Confidence scores для каждой категории рас.
            - 'indian': показатель уверенности для индийской национальности.
            - 'asian': показатель уверенности для азиатской национальности.
            - 'latino hispanic': показатель уверенности для латиноамериканской/испаноязычной национальности.
            - 'black': показатель уверенности для чернокожей национальности.
            - 'middle eastern': показатель доверия к ближневосточной этнической принадлежности.
            - 'white': показатель доверия к белой этнической принадлежности.
        """
        result = [{
            "region": {
                'x': 0,
                'y': 0,
                'w': 0,
                'h': 0
            },
            "age": 0,
            "face_confidence": 0,
            "dominant_gender": "",
            "gender": {
                'Man': 0,
                'Woman': 0
            },
            "dominant_emotion": "",
            "emotion": {
                'sad': 0,
                'angry': 0,
                'surprise': 0,
                'fear': 0,
                'happy': 0,
                'disgust': 0,
                'neutral': 0
            },
            "dominant_race": "",
            "race": {
                'indian': 0,
                'asian': 0,
                'latino hispanic': 0,
                'black': 0,
                'middle eastern': 0,
                'white': 0
            }
        }]


        try:
            result = DeepFace.analyze(
                img_path = image, 
                detector_backend = 'retinaface',
                silent = True,
                actions = ['age', 'gender', 'race', 'emotion']
            )
        except:
            print("На изображении нет лиц. Используйте изображения с лицами")
            return result

        if self.save_results:
            with open(os.path.join(self.save_path, f'analyze_{id}.json'), 'w') as f:
                json.dump(result, f)

        return result
    

