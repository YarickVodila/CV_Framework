# CV Framework
> Фреймворк компьютерного зрения для задач, детекции, сегментации, классификации, идентификации людей, анализа настроений, определения пола, возраста, расы и классификации действий

## Содержание

1. [Описание](#описание)
2. [Установка](#установка)
3. [Документация](#документация)
    - [Методы класса Pipeline](#методы-класса-pipeline)
    - [Методы класса ActionClassification](#методы-класса-actionclassification)
    - [Методы класса EfficientNetClassification](#методы-класса-efficientnetclassification)
    - [Методы класса YoloDetection](#методы-класса-yolodetection)
    - [Методы класса YoloSegmentation](#методы-класса-yolosegmentation)
    - [Методы класса FaceAnalyze](#методы-класса-faceanalyze)
    - [Методы класса FaceRecognition](#методы-класса-facerecognition)

## Описание



## Установка

Перед использование фреймворка необходимо


1. **Клонировать репозиторий**

```cmd
git clone https://github.com/YarickVodila/CV_Framework.git
```

2. **Установить необходимые библиотеки и зависимости**
> [!IMPORTANT]
> Для использования моделей `Torch` на `cuda` используйте официальную документацию [PyTorch](https://pytorch.org/get-started/locally/)
Иначе модели будут использоваться на `cpu`.

```py
pip install -r requirements.txt
```

## Документация

Основным классом в фреймворке является класс [Pipeline](framework/pipeline.py). Пользователь использует объект данного класса для решения задач.

### Методы класса Pipeline


```py
def add_pipe(self, name, **kwargs):
```

>Метод добавления компонента в pipeline

**Параметры:**
- `name` (str): Название компонента. Допустимые значения: `classification`, `detection`, `segmentation`, `face_analyze`, `face_recognition` `action_classification`
- `**kwargs` : Дополнительные аргументы для компонента.


<hr/>
<br/>



```py
def del_pipe(self, name):
```
>Метод удаления компонента Pipeline

**Параметры**:
- `name` (str): Название компонента. Допустимые значения: `classification`, `detection`, `segmentation`, `face_analyze`, `face_recognition`, `action_classification`


<hr/>
<br/>



```py
def predict(self, image: Union[str, np.ndarray, List], **kwargs) -> Dict:
```

>Метод предсказания компонентов Pipeline

**Параметры**:
- `image` (Union[str, np.ndarray, List]): Предсказываемое изображение или список изображений.
- `**kwargs`: Дополнительные аргументы 

**Возвращаемое значение**:
- `result` (Dict): Словарь с предсказаниями

**Пример**:

```py
result = pipeline.predict(image)
```

<hr/>
<br/>



```py
def save_pipeline(self, path_name: str = 'pipeline'):
```
>Метод сохранения компонентов Pipeline

**Параметры**:
- `path_name` (str): Название папки `#0969DA`, в которую будут сохранены компоненты Pipeline
<hr/>
<br/>



```py
def load_pipeline(self, path_name: str = 'pipeline'):
```

>Метод загрузки компонентов Pipeline

**Параметры**:
- `path_name` (str): Название папки, в которой хранятся компоненты Pipeline


<hr/>
<br/>


### Методы класса ActionClassification

```py
def __init__(self, input_shape: tuple = (10, 240, 240, 3), num_classes: int = 2):
```

>Инициализация модели классификации действий.

**Параметры**:
- `input_shape` (tuple, необязательный): форма входных данных, по умолчанию (10, 240, 240, 3)
    - 10 - количество кадров за 1 раз
    - 240 - высота изображения
    - 240 - ширина изображения
    - 3 - количество каналов (RGB)
- `num_classes` (int, необязательный): количество классов, по умолчанию 2


**Примечание**:
- Если входная форма не соответствует вашей задаче, измените параметр "input_shape"
- Если выходная форма не соответствует вашей задаче, измените параметр "num_classes"
- Если параметр "input_shape" не является кортежем из 4 элементов, будет выброшено исключение ValueError

<hr/>
<br/>


```py
def conv_batchnorm_relu(self, x, filters, kernel_size, strides=1):
```

>Применяет последовательность операций свёртки, batch normalization и ReLU к входному тензору.

**Аргументы**:
- `x` (tf.Tensor): Входной тензор.
- `filters` (int): Количество фильтров в свёрточном слое.
- `kernel_size` (int): Размер фильтра свёрточном слоя.
- `strides` (int, optional): Степень сдвига свёрточном слоя. По умолчанию равен 1.

**Возвращает**:
- `tf.Tensor`: Выходной тензор после применения операций свёртки, batch normalization и ReLU.

<hr/>
<br/>


```py
def identity_block(self, tensor, filters):
```

>Производит блок идентичности в нейронной сети.
        
**Аргументы**:
- `tensor` (tf.Tensor): Входной тензор.
- `filters` (int): Количество фильтров.

**Возвращает**:
- `tf.Tensor`: Выходной тензор после прохождения блока идентичности. 

<hr/>
<br/>


```py
def projection_block(self, tensor, filters, strides):
```

>Создает блок проекции в нейронной сети.

**Аргументы**:
- `tensor` (tf.Tensor): Входной тензор.
- `filters` (int): Количество фильтров.
- `strides` (int): Шаг сдвига.

**Возвращает**:
- `tf.Tensor`: Выходной тензор после прохождения блока проекции.

<hr/>
<br/>


```py
def resnet_block(self, x, filters, reps, strides):
```

>Создает блок ResNet в нейронной сети.

**Аргументы**:
- `x` (tf.Tensor): Входной тензор.
- `filters` (int): Количество фильтров.
- `reps` (int): Количество повторений.
- `strides` (int): Шаг сдвига.

**Возвращает**:
- `tf.Tensor`: Выходной тензор после прохождения блока ResNet.

<hr/>
<br/>


```py
def predict(self, x):
```

>Предсказывает выход для данного входного изображения с помощью обученной модели.

`Аргументы`:
- `x` (numpy.ndarray): Массив с входными данными для предсказания.

`Возвращает`:
- `numpy.ndarray`: Массив с предсказанными значениями.

<hr/>
<br/>



### Методы класса EfficientNetClassification

```py
def __init__(self, model_size:str = "s", transfer_learning:bool = True, layers: list | tuple = None, device: str = 'cuda'):
```

>Инициализирует экземпляр класса EfficientNetClassification.
        
**Аргументы**:
- `model_size` (str, optional): Размер модели EfficientNet. Должен быть одним из "s", "m" или "l". По умолчанию "s".
- `transfer_learning` (bool, optional): Флаг использования трансферного обучения. По умолчанию `True`.
- `layers` (list | tuple, optional): Порядок слоев в модели для классификации. По умолчанию `None`.
- `device` (str, optional): Устройство для обучения. По умолчанию `cuda`.

**Исключения**:
- `AssertionError`: Если model_size не "s", "m" или "l".

**Примечания**:
- model_size определяет размер загружаемой модели EfficientNet.
- Если transfer_learning установлен в True, веса модели замораживаются, и обучаются только веса классификатора.
- Если предоставлены слои, к модели для классификации добавляются новые слои.
- Если слои не предоставлены, и is_trainable установлен в True, в модели два нейрона на выходе с стандартными слоями, предложенными авторами модели.
- Если не предоставлены ни слои, ни is_trainable, модель имеет 1000 классов на выходе, как в наборе данных ImageNet1K.
- Модель перемещается на указанное устройство.

<hr/>
<br/>


```py
def resize_img(self, img:np.array) -> torch.tensor: 
```

>"Ресайз" изображения до заданных размеров (480, 480).
        
**Аргументы**:
- `img`: np.array Исходное изображение

**Возвращает**:
- `torch.tensor` "Ресайзнутое" изображение в формате torch.tensor
<hr/>
<br/>


```py
def predict(self, images: Union[str, np.ndarray, List], return_probs: bool = False) -> torch.tensor:
```

>Метод предсказывает классы для переданных изображений с помощью модели классификации.

**Аргументы**:
- `images` (Union[str, np.ndarray, List]): Массив изображений, которые нужно предсказать. Может быть строка, numpy.ndarray или список строк или numpy.ndarray.
- `return_probs` (bool, optional): Если `True`, то возвращает вероятности для каждого класса. По умолчанию `False`. 

**Возвращает**:
- `torch.tensor`: Массив предсказанных классов или вероятностей для каждого класса, в зависимости от значения `return_probs`.

<hr/>
<br/>

### Методы класса YoloDetection

```py
def __init__(self, model_size:str, yolo_result=True, **kwargs_for_predict):
```

>Конструктор класса YoloDetection.

**Параметры**:
- `model_size` (str): Размер модели YOLO. Допустимые значения: 'n', 's', 'm', 'l', 'x'.
- `yolo_result` (bool): Опция для возврата результата в формате YOLO ultralytics. По умолчанию True
- `**kwargs_for_predict`: Дополнительные аргументы для метода predict модели.

**Примеры использования**:
```py
detection = YoloDetection(model_size='m', verbose=False, conf=0.6)
```

<hr/>
<br/>


```py
def predict(self, images:str | list | tuple):
```

>Предсказывает объекты на изображениях.

**Параметры**:
- `images` (str | list | tuple): Путь к изображению или список путей к изображениям для предсказания.

**Возвращает**:
- Если `yolo_result` = `True`, возвращает список результатов YOLO (result[i].boxes подробнее смотреть по ссылке https://docs.ultralytics.com/ru/modes/predict/#boxes).
- Если `yolo_result` = `False`, возвращает словарь с информацией о предсказанных объектах на каждом изображении. `Пример ниже`
        
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
**Примеры использования**:
```py
results = detection.predict(images=["image1.jpg", "image2.jpg"], yolo_result=True)
results_custom = detection.predict("image3.jpg", yolo_result=False)
```

<hr/>
<br/>

### Методы класса YoloSegmentation

```py
def __init__(self, model_size:str, yolo_result=True, **kwargs_for_predict):
```

>Конструктор класса YoloSegmentation.

**Параметры**:
- `model_size` (str): Размер модели YOLO. Допустимые значения: 'n', 's', 'm', 'l', 'x'.
- `yolo_result` (bool): Опция для возврата результата в формате YOLO ultralytics. По умолчанию True

- `**kwargs_for_predict`: Дополнительные аргументы для метода predict модели.

**Примеры использования**:
```py
segmentation = YoloSegmentation(model_size='m', verbose=False, conf=0.6)
```

<hr/>
<br/>



```py
def predict(self, images:str | list | tuple):
```

>Предсказывает объекты на изображениях.

**Параметры**:
- `images` (str | list | tuple): Путь к изображению, список путей к изображениям для предсказания, массив np.array (Height x Wight x Channel) изображений. 
    

**Возвращает**:
- Если `yolo_result` = `True`, возвращает список результатов YOLO (result[i].boxes подробнее смотреть по ссылке https://docs.ultralytics.com/ru/modes/predict/#boxes).
- Если `yolo_result` = `False`, возвращает словарь с информацией о предсказанных объектах на каждом изображении. Пример ниже
        
```py
{
    'файл.png': {
        "classes": np.array([0]),
        "str_classes": ["person"],
        'masks': [array([[      502.5,         105],
                [     500.62,      106.87],
                [     476.25,      106.87],
                ...,
                [     571.88,      106.87],
                [     553.12,      106.87],
                [     551.25,         105]], dtype=float32)],
        "bboxs": np.array([[550.25, 690.15, 825.81, 1016.4]], dtype=float32),
        "conf": np.array([0.7562], dtype=float32)
    }
}
```
        
**Примеры использования**:
```py
results = segmentation.predict(images=["image1.jpg", "image2.jpg"], yolo_result=True)
results_custom = segmentation.predict("image3.jpg", yolo_result=False)
```

<hr/>
<br/>

### Методы класса FaceAnalyze

```py
def __init__(self, save_results:bool = False, save_path:str = ""):
```

**Параметры**:
- `save_results` (bool, optional): Необходимо ли сохранять результат анализа в файл `"analyze_id.json"`. По умолчанию False.
- `save_path` (str, optional): Путь до директории, в которую необходимо сохранять результат анализа. По умолчанию "".


<hr/>
<br/>

```py
def predict(self, image: Union[str, np.ndarray], id:int = 0, **kwargs):
```


**Параметры**:
- `image` (Union[str, np.ndarray]): Путь до исходного изображения или открытое изображение 
- `id` (int, optional): Идентификатор изображения. По умолчанию 0.

**Возвращает**:
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

<hr/>
<br/>

### Методы класса FaceRecognition

```py
def __init__(self, path_embeddings:str = ""):
```
**Параметры**:
- `path_embeddings` (str, optional): Путь до `embeddings.parquet` файла. По умолчанию "".

<hr/>
<br/>


```py
def l2_normalize(self, x:np.ndarray):
```

>Нормализация вектора x по L2
        
**Параметры**:
- `x` (np.ndarray): Массив numpy

**Возвращает**:
- `np.ndarray`: Нормализованный вектор по L2

<hr/>
<br/>


```py
def distance(self, embedding1:np.ndarray, embedding2:np.ndarray):
```

>Метод для подсчёта Евклидова расстояния между двумя эмбедингами 

**Параметры**:
- `embedding1` (np.ndarray(512,)): Массив numpy с эмбедингом 1-го лица
- `embedding2` (np.ndarray(512,)): Массив numpy с эмбедингом 2-го лица

**Возвращает**:
- `float64`: Евклидово расстояние между эмбедингами

<hr/>
<br/>


```py
def predict(self, image: Union[str, np.ndarray], append_new_person:bool = True, **kwargs):
```

>Метод для анализа изображений путём получения эмбедингов и проверки евклидова расстояния между эмбедингами в базе

**Параметры**:
- `image`: (np.array or str). Путь до исходного изображения или открытое изображение
- `append_new_person`: (bool). Необходимо ли добавлять новых людей на изображении

**Возвращает**:
- `dict`: Словарь с ключами `id` и `coords`, где id - уникальный идентификатор человека, coords - координаты лица в виде словаря
    с ключами x, y, w, h.

    - x, y - координаты левого верхнего угла рамки с лицом

    - w, h - координаты ширины и высоты рамки с лицом

<hr/>
<br/>