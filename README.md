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

**Метод добавления компонента в pipeline**

**Параметры:**
- `name` (str): Название компонента. Допустимые значения: `classification`, `detection`, `segmentation`, `face_analyze`, `face_recognition` `action_classification`
- `**kwargs` : Дополнительные аргументы для компонента.


<hr/>
<br/>

```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```


```py
def del_pipe(self, name):
```
**Метод удаления компонента Pipeline**

**Параметры**:
- `name` (str): Название компонента. Допустимые значения: `classification`, `detection`, `segmentation`, `face_analyze`, `face_recognition`, `action_classification`


<hr/>
<br/>



```py
def predict(self, image: Union[str, np.ndarray, List], **kwargs) -> Dict:
```

**Метод предсказания компонентов Pipeline**

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
**Метод сохранения компонентов Pipeline**

**Параметры**:
- `path_name` (str): Название папки `#0969DA`, в которую будут сохранены компоненты Pipeline
<hr/>




```py

```

<hr/>
<br/>


### Методы класса ActionClassification


### Методы класса EfficientNetClassification


### Методы класса YoloDetection


### Методы класса YoloSegmentation


### Методы класса FaceAnalyze


### Методы класса FaceRecognition