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

```

>

<hr/>
<br/>


### Методы класса YoloDetection

```py

```

>

<hr/>
<br/>

### Методы класса YoloSegmentation

```py

```

>

<hr/>
<br/>

### Методы класса FaceAnalyze

```py

```

>

<hr/>
<br/>

### Методы класса FaceRecognition

```py

```

>

<hr/>
<br/>