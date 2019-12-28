# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:13:29 2019

@author: 潘浩宇
"""

import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']


token_index = {}
for sample in samples:  # sentens in list'samples'
    for word in sample.split(): # word in list'sentens'
        if word not in token_index: # if not appeared before, add to index
            token_index[word] = len(token_index) + 1 
            # print(token_index)
            
max_len = 10 #sample pre 10 words

result = np.zeros(shape=(len(samples),
                  max_len,
                  max(token_index.values())+1)) # Build a ndarray filled by 0

for i,samples in enumerate(samples):
    for j,word in list(enumerate(sample.split()))[:max_len]:
        index = token_index.get(word)
        result[i, j, index] = 1
        