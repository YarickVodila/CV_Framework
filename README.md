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


### Методы класса ActionClassification


### Методы класса EfficientNetClassification


### Методы класса YoloDetection


### Методы класса YoloSegmentation


### Методы класса FaceAnalyze


### Методы класса FaceRecognition