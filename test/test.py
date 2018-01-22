# -*- coding: utf-8 -*-

import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD

# ������ seed ��� ������������� �����������
numpy.random.seed(42)

# ��������� ������
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# ������ ����-�������
batch_size = 32
# ���������� ������� �����������
nb_classes = 10
# ���������� ���� ��� ��������
nb_epoch = 25
# ������ �����������
img_rows, img_cols = 32, 32
# ���������� ������� � �����������: RGB
img_channels = 3

# ����������� ������
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# ����������� ����� � ���������
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# ������� ���������������� ������
model = Sequential()
# ������ ���������� ����
model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=(32, 32, 3), activation='relu'))
# ������ ���������� ����
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# ������ ���� ����������
model.add(MaxPooling2D(pool_size=(2, 2)))
# ���� ������������� Dropout
model.add(Dropout(0.25))

# ������ ���������� ����
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# ��������� ���������� ����
model.add(Conv2D(64, (3, 3), activation='relu'))
# ������ ���� ����������
model.add(MaxPooling2D(pool_size=(2, 2)))
# ���� ������������� Dropout
model.add(Dropout(0.25))
# ���� �������������� ������ �� 2D ������������� � �������
model.add(Flatten())
# ������������ ���� ��� �������������
model.add(Dense(512, activation='relu'))
# ���� ������������� Dropout
model.add(Dropout(0.5))
# �������� ������������ ����
model.add(Dense(nb_classes, activation='softmax'))

# ������ ��������� �����������
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
# ������� ������
model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_split=0.1,
              shuffle=True,
              verbose=2)

# ��������� �������� �������� ������ �� �������� ������
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))