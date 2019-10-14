#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/14 16:23
# @Author : zui
# @File : test.py

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

W1 = np.array([[0.15, 0.20], [0.25, 0.30]])
W2 = np.array([[0.40, 0.45], [0.50, 0.55]])
b1 = np.array([[0.35], [0.35]])
b2 = np.array([[0.90], [0.60]])
input = np.array([[0.05], [0.10]])
input2 = np.array([[0.59326999], [0.59688438]])

# print(np.matmul(W1, input))
# print(sigmoid(np.matmul(W1, input)+b1))
m1 = sigmoid(np.matmul(W1, input)+b1)
print(m1)

print(np.transpose(m1))