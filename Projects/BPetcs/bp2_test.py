#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/14 15:03
# @Author : zui
# @File : bp2_test.py

from bp2 import NeuralNetwork
import numpy as np

nn = NeuralNetwork([2, 2, 1], 'tanh')
temp = [[0, 0], [0, 1], [1, 0], [1, 1]]
X = np.array(temp)
y = np.array([0, 1, 1, 0])
nn.fit(X, y)
for i in temp:
    print(i, nn.predict(i))