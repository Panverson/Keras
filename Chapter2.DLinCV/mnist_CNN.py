# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 18:22:04 2019

@author: 潘浩宇
"""
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers
from keras import models

# Build CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# print(model.summary())

# Add classifier
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
# print(model.summary())

# Preprocess data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize data
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Compile model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Input data
model.fit(train_images, train_labels, epochs=5, batch_size=64)

'''
Result:
    loss:0.0435
    acc:0.9884(1.06%👆)
'''