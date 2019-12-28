# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:47:54 2019

@author: 潘浩宇
"""

from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

tokenizer = Tokenizer(samples)
tokenizer.fit_on_texts(samples)

sequences = tokenizer.texts_to_sequences(samples)

result = tokenizer.texts_to_matrix(samples)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))