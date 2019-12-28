# -*- coding: utf-8 -*-

from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models, layers

# Read data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Build network
'''
architeture:
    2 FC layers(1 input layer, 1 output layer)
'''
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28, )))
# Softmax layer: Return 10 probability values whose sum is 1
network.add(layers.Dense(10, activation='softmax'))

# Compile
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Normalization
'''
Reshaping the size of input data to adjust the format of the network,
and zooming all values into interval [0,1]
'''
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

'''
Result:
    test_loss = 0.7364
    test_acc = 0.978
'''