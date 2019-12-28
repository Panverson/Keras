# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:57:20 2019

@author: 潘浩宇
"""

from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
        num_words= 10000)
