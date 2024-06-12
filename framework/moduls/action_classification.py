# from tensorflow.keras import Sequential
from tensorflow import keras
# from tensorflow.keras import layers

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, GlobalAvgPool2D
from tensorflow.keras.layers import Add, ReLU, Dense, LSTM, TimeDistributed
from tensorflow.keras import Model


import tensorflow as tf

class ActionClassification:
    def __init__(self, input_shape: tuple = (10, 240, 240, 3), num_classes: int = 2):
        """
        Инициализация модели классификации действий.

        Параметры:
        - input_shape (tuple, необязательный): форма входных данных, по умолчанию (10, 240, 240, 3)
            - 10 - количество кадров за 1 раз
            - 240 - высота изображения
            - 240 - ширина изображения
            - 3 - количество каналов (RGB)
        - num_classes (int, необязательный): количество классов, по умолчанию 2


        Примечание:
        - Если входная форма не соответствует вашей задаче, измените параметр "input_shape"
        - Если выходная форма не соответствует вашей задаче, измените параметр "num_classes"
        - Если параметр "input_shape" не является кортежем из 4 элементов, будет выброшено исключение ValueError
        """
        input = Input(shape=input_shape)

        x = self.conv_batchnorm_relu(input, filters = 64, kernel_size = 7, strides = 2)
        x = TimeDistributed(MaxPool2D(pool_size = 3, strides = 2))(x)
        x = self.resnet_block(x, filters = 64, reps = 3, strides = 1)
        x = self.resnet_block(x, filters = 128, reps = 4, strides = 2)
        x = self.resnet_block(x, filters = 256, reps = 6, strides = 2)
        # x = resnet_block(x, filters=512, reps =3, strides=2)
        x = TimeDistributed(GlobalAvgPool2D())(x)
        x = LSTM(128, return_sequences=True, dropout = 0.2)(x)
        x = LSTM(64, return_sequences=True, dropout = 0.2)(x)
        x = LSTM(32, return_sequences=False, dropout = 0.1)(x)


        optimizer = tf.keras.optimizers.Adam(0.0001)

        if num_classes == 2:
            output = Dense(1, activation ='sigmoid')(x)
            self.model = Model(inputs=input, outputs=output)
            self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[keras.metrics.Precision(), keras.metrics.AUC()])
        else:
            output = Dense(num_classes, activation ='softmax')(x)
            self.model = Model(inputs=input, outputs=output)
            self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[keras.metrics.CategoricalAccuracy()])

        if len(input_shape) == 4:
            print(f"""
            Модель на вход получает форму {input_shape} где {input_shape[0]} - количество кадров за 1 раз, {input_shape[1]}  - высота изображения, {input_shape[2]}  - ширина изображения, {input_shape[3]}  - количество каналов (RGB)
            
            Выход модели: {output.shape} 
            
            Если входная форма не соответствует вашей задаче измените параметр "input_shape"
            Если выходная форма не соответствует вашей задаче измените параметр "num_classes"

            """)
        else:
            raise ValueError(f"""
                            "input_shape" должен быть формой из 4 значений например (10, 240, 240, 3), где
                             
                            10 - количество кадров за 1 раз,
                            240 - высота изображения,
                            240 - ширина изображения,
                            3 - количество каналов (RGB)
                            """)

    def conv_batchnorm_relu(self, x, filters, kernel_size, strides=1):
        """
        Применяет последовательность операций свёртки, batch normalization и ReLU к входному тензору.
    
        Аргументы:
            x (tf.Tensor): Входной тензор.
            filters (int): Количество фильтров в свёрточном слое.
            kernel_size (int): Размер фильтра свёрточном слоя.
            strides (int, optional): Степень сдвига свёрточном слоя. По умолчанию равен 1.
    
        Возвращает:
            tf.Tensor: Выходной тензор после применения операций свёртки, batch normalization и ReLU.
        """
        x = TimeDistributed(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding = 'same'))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(ReLU())(x)
        
        return x


    def identity_block(self, tensor, filters):
        """
        Производит блок идентичности в нейронной сети.
        
        Аргументы:
            tensor (tf.Tensor): Входной тензор.
            filters (int): Количество фильтров.
        
        Возвращает:
            tf.Tensor: Выходной тензор после прохождения блока идентичности. 
        """
        
        x = self.conv_batchnorm_relu(tensor, filters = filters, kernel_size = 1, strides = 1)
        x = self.conv_batchnorm_relu(x, filters = filters, kernel_size = 3, strides = 1)
        x = TimeDistributed(Conv2D(filters = 4*filters, kernel_size = 1, strides = 1))(x)
        x = TimeDistributed(BatchNormalization())(x)
        
        x = Add()([tensor,x])    #skip connection
        x = ReLU()(x)
        
        return x



    def projection_block(self, tensor, filters, strides):
        """
        Создает блок проекции в нейронной сети.

        Аргументы:
            tensor (tf.Tensor): Входной тензор.
            filters (int): Количество фильтров.
            strides (int): Шаг сдвига.

        Возвращает:
            tf.Tensor: Выходной тензор после прохождения блока проекции.
        """
        
        #left stream
        x = self.conv_batchnorm_relu(tensor, filters = filters, kernel_size = 1, strides = strides)
        x = self.conv_batchnorm_relu(x, filters = filters, kernel_size = 3, strides = 1)
        x = TimeDistributed(Conv2D(filters = 4*filters, kernel_size = 1, strides = 1))(x)
        x = TimeDistributed(BatchNormalization())(x)
        
        #right stream
        shortcut = TimeDistributed(Conv2D(filters = 4*filters, kernel_size = 1, strides = strides))(tensor)
        shortcut = TimeDistributed(BatchNormalization())(shortcut)
        
        x = Add()([shortcut,x])    #skip connection
        x = ReLU()(x)
        
        return x


    def resnet_block(self, x, filters, reps, strides):
        """
        Создает блок ResNet в нейронной сети.

        Аргументы:
            x (tf.Tensor): Входной тензор.
            filters (int): Количество фильтров.
            reps (int): Количество повторений.
            strides (int): Шаг сдвига.

        Возвращает:
            tf.Tensor: Выходной тензор после прохождения блока ResNet.
        """
        
        x = self.projection_block(x, filters, strides)
        for _ in range(reps-1):
            x = self.identity_block(x,filters)
            
        return x
    
    def predict(self, x):
        """
        Предсказывает выход для данного входного изображения с помощью обученной модели.

        Аргументы:
            x (numpy.ndarray): Массив с входными данными для предсказания.

        Возвращает:
            numpy.ndarray: Массив с предсказанными значениями.
        """
    
        return self.model.predict(x)