from deepface import DeepFace
import os
import numpy as np
import pandas as pd

from typing import Any, Dict, List, Tuple, Union, Optional


class FaceRecognition:
    def __init__(self, path_embeddings:str = ""):
        """

        Args:
            path_embeddings (str, optional): Путь до `embeddings.parquet` файла. По умолчанию "".
        """

        self.path_embeddings = path_embeddings

        try:
            self.data = pd.read_parquet(os.path.join(self.path_embeddings, 'embeddings.parquet'), engine='pyarrow')
        except:
            print(f"Не найден файл embeddings.parquet в директории '{self.path_embeddings}' !!! Будет создан новый файл")

            self.data = pd.DataFrame({"id":[], "embedding": []})
            self.data.to_parquet(os.path.join(self.path_embeddings, 'embeddings.parquet'), engine='pyarrow')

        # print(data.empty)

    def __name__(self):
        return 'face_recognition'
    
    def l2_normalize(self, x:np.ndarray):
        """
        Нормализация вектора x по L2
        
        Args:
            x (np.ndarray): Массив numpy

        Returns:
            np.ndarray: Нормализованный вектор по L2
        """
        x = np.array(x)
        return x / np.sqrt(np.sum(np.multiply(x, x)))


    def distance(self, embedding1:np.ndarray, embedding2:np.ndarray):
        """
        Метод для подсчёта Евклидова расстояния между двумя эмбедингами 

        Args:
            embedding1 (np.ndarray(512,)): Массив numpy с эмбедингом 1-го лица
            embedding2 (np.ndarray(512,)): Массив numpy с эмбедингом 2-го лица

        Returns:
            float64: Евклидово расстояние между эмбедингами
        """
        return np.sqrt(np.sum(np.square(np.subtract(embedding1, embedding2))))

    
    def predict(self, image: Union[str, np.ndarray], append_new_person:bool = True, **kwargs):
        """ 
        Метод для анализа изображений путём получения эмбедингов и проверки евклидова расстояния между эмбедингами в базе

        Args:
            image: (np.array or str). Путь до исходного изображения или открытое изображение
            append_new_person: (bool). Необходимо ли добавлять новых людей на изображении
        
        Returns:
            dict: Словарь с ключами `id` и `coords`, где id - уникальный идентификатор человека, coords - координаты лица в виде словаря
            с ключами x, y, w, h.

            x, y - координаты левого верхнего угла рамки с лицом

            w, h - координаты ширины и высоты рамки с лицом
        """

        embeddings = []
        coords_face = []
        labels = []
        detections_peoples = {'id':[], 'coords':[]}

        # Получаем эмбединги всех лиц на изображении + координты лица + порог уверенности нахождения лица
        try:
            embedding_objs = DeepFace.represent(img_path = image, model_name = "ArcFace", detector_backend = "retinaface", enforce_detection = True)
        except:
            print("На изображении нет лиц. Используйте изображения с лицами")

            return detections_peoples

        # Сохраняем полученные результаты эмбедингов и координат
        for emb in embedding_objs:
            embeddings.append(emb['embedding']) # Эмбединг лица
            coords_face.append(emb['facial_area']) # словарь с координатами лица в формате x, y - координаты левого верхнего угла, w, h - ширина и высота


        # Если мы ходим добавить нового человека в базу данных
        if append_new_person == True:
            if not self.data.empty: # Если база данных не пустая
                for i, emb in enumerate(embeddings):
                    self.data['distance'] = self.data['embedding'].apply(lambda x: self.distance(x, self.l2_normalize(emb)))

                    # Найденный человек и его id в формате [id]
                    person_id = self.data.loc[(self.data['distance']<1.15) & (self.data['distance']==self.data['distance'].min())]['id'].to_list()
                    self.data.drop(columns=['distance'], inplace=True)
                    
                    # если человека в базе нет
                    if not person_id:
                        labels.append(f"person_{self.data['id'].max()+1}")
                        detections_peoples['id'].append(f"person_{self.data['id'].max()+1}")
                        detections_peoples['coords'].append(coords_face[i])

                        new_row = pd.DataFrame({'id': [self.data['id'].max()+1], 'embedding': [self.l2_normalize(emb)]})
                        self.data = pd.concat([self.data, new_row], ignore_index=True)

                        
                        # display(data)

                    # если человека в базе есть
                    else:
                        labels.append(f"person_{person_id[0]}")
                        detections_peoples['id'].append(f"person_{person_id[0]}")
                        detections_peoples['coords'].append(coords_face[i])

            else: # Если база данных пустая
                for i, emb in enumerate(embeddings):
                    new_row = pd.DataFrame({'id': [i], 'embedding': [self.l2_normalize(emb)]})
                    self.data = pd.concat([self.data, new_row])
                    
                    labels.append(f"person_{i}")
                    detections_peoples['id'].append(f"person_{i}")
                    detections_peoples['coords'].append(coords_face[i])

                self.data['id'] = self.data['id'].astype(int)
                # display(data)

            # Сохраняем изменения
            self.data.to_parquet(os.path.join(self.path_embeddings, 'embeddings.parquet'), index=False, engine='pyarrow')

        else: # Без добавления в базу данных
            if not self.data.empty: # Если база данных не пустая
                
                for i, emb in enumerate(embeddings):
                    self.data['distance'] = self.data['embedding'].apply(lambda x: self.distance(x, self.l2_normalize(emb)))
                    # Найденный человек и его id в формате [id]
                    person_id = self.data.loc[(self.data['distance']<1.15) & (self.data['distance']==self.data['distance'].min())]['id'].to_list()
                    # display(data.loc[(data['distance']<4.15) & (data['distance']==data['distance'].min())])
                    # display(data)
                    self.data.drop(columns=['distance'], inplace=True)
                    
                    # если человека в базе нет
                    if not person_id:
                        labels.append(f"undefind_{i}")
                        detections_peoples['id'].append(f"undefind_{i}")
                        detections_peoples['coords'].append(coords_face[i])

                    # если человека в базе есть
                    else:
                        # print(f'Человек в базе есть {person_id[0]}')
                        labels.append(f"person_{person_id[0]}")
                        detections_peoples['id'].append(f"person_{person_id[0]}")
                        detections_peoples['coords'].append(coords_face[i])

            else: # Если база данных пустая
                for i, emb in enumerate(embeddings):
                    labels.append(f"undefind_{i}")
                    detections_peoples['id'].append(f"undefind_{i}")
                    detections_peoples['coords'].append(coords_face[i])

        return detections_peoples
