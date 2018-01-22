# -*- coding: utf-8 -*-

# 7 минут

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from os import listdir
from os.path import isfile
from os.path import join as joinpath

datagen = ImageDataGenerator(
                            rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            rescale=1./255,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest')
X=np.zeros((0,100,100,3))
y=np.zeros((0,1))

for i in listdir("GT"):
    if isfile(joinpath("GT", i)):
        img =image.load_img("../watchtophoto/GT/"+i) # this is a PIL image
        x = image.img_to_array(img) # this is a Numpy array with shape (3, 150, 150)
        x = np.expand_dims(x, axis=0)
        X=np.append(X,x/255,axis=0)
        y=np.append(y,1)

for i in listdir("BT"):
    if isfile(joinpath("BT", i)):
        img =image.load_img("../watchtophoto/BT/"+i) # this is a PIL image
        x = image.img_to_array(img) # this is a Numpy array with shape (3, 150, 150)
        x = np.expand_dims(x, axis=0)
        X=np.append(X,x/255,axis=0)
        y = np.append(y, 0)


X_test=np.zeros((0,100,100,3))
y_test=[]

for i in listdir("GTT"):
    if isfile(joinpath("GTT", i)):
        img =image.load_img("../watchtophoto/GTT/"+i) # this is a PIL image
        x = image.img_to_array(img) # this is a Numpy array with shape (3, 150, 150)
        x = np.expand_dims(x, axis=0)
        X_test=np.append(X_test,x/255,axis=0)

        y_test=y_test+['Хороший']
for i in listdir("BTT"):
    if isfile(joinpath("BTT", i)):
        img =image.load_img("../watchtophoto/BTT/"+i) # this is a PIL image
        x = image.img_to_array(img) # this is a Numpy array with shape (3, 150, 150)
        x = np.expand_dims(x, axis=0)
        X_test=np.append(X_test,x/255,axis=0)
        y_test = y_test + ['Плохой']
#fh_o = open('X.txt', 'wb')
#np.savetxt(fh_o, X,fmt=' %.18e', delimiter = '\n' )
# Размер мини-выборки
batch_size = 36
# Количество классов изображений
nb_classes = 2
# Количество эпох для обучения
nb_epoch = 20

# Размер изображений
img_rows, img_cols = 100, 100
# Количество каналов в изображении: RGB
img_channels = 3
# Создаем последовательную модель
Y=np_utils.to_categorical(y,nb_classes)
model = Sequential()
# Первый сверточный слой
model.add(Conv2D(8, (3, 3), padding='same', input_shape=(100, 100, 3), activation='relu')) #попробовать первые слои 5 на 5
# Второй сверточный слой
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
# Первый слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# Слой регуляризации Dropout
model.add(Dropout(0.25))

model.add(Conv2D(8, (3, 3), padding='same', input_shape=(100, 100, 3), activation='relu'))

model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))
# Слой регуляризации Dropout
model.add(Dropout(0.25))


# Третий сверточный слой
model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
# Четвертый сверточный слой
model.add(Conv2D(8, (3, 3), activation='relu'))
# Второй слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# Слой регуляризации Dropout
model.add(Dropout(0.25))
# Слой преобразования данных из 2D представления в плоское
model.add(Flatten())
# Полносвязный слой для классификации
model.add(Dense(512, activation='relu'))
# Слой регуляризации Dropout
model.add(Dropout(0.5))
# Выходной полносвязный слой
model.add(Dense(nb_classes, activation='softmax'))


# Задаем параметры оптимизации
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
# Обучаем модель
model.fit(X, Y, batch_size=batch_size, epochs=nb_epoch, validation_split=0.1, shuffle=True, verbose=2)
scores = model.predict(X_test, verbose=0)

g = 0
corrector = 0
for i in y_test:
    if scores[g,0] > scores[g,1] and i == 'Хороший':
        corrector += 1
    elif scores[g,0] < scores[g,1] and i == 'Плохой':
        corrector += 1
    print("Прогноз:", scores[g], " На самом деле", i, "\n")
    g += 1
print('\n', 'Точность прогноза составила : ', round((corrector/g)*100,3), '%')
model.save_weights("Model.txt")