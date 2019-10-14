#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/14 11:55
# @Author : zui
# @File : bp.py

import numpy as np

# sigmoid函数
def sigmoid(x):
    return 1/(1+np.exp(-x))

# sigmoid函数的导数
def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))

class BPnetwork(object):
    def __init__(self):
        self.alpha = 0.5    # 学习率
        self.W1 = np.array([[0.15, 0.20], [0.25, 0.30]])   #输入层到卷积层的系数矩阵
        self.b1 = np.array([[0.35], [0.35]])
        self.W2 = np.array([[0.40, 0.45], [0.50, 0.55]])
        self.b2 = np.array([[0.90], [0.60]])
        self.input = np.array([[0.05], [0.10]])
        self.output = np.array([[0.05], [0.9]])

    def feedfoward(self):
        Z1 = np.matmul(self.W1, self.input) + self.b1
        A1 = sigmoid(Z1)
        Z2 = np.matmul(self.W2, A1) + self.b2
        A2 = sigmoid(Z2)
        return Z1, A1, Z2, A2

    def propogate(self):
        z1, a1, z2, a2 = self.feedfoward()
        delta2 = -(self.output - a2) * sigmoid_derivative(z2)
        self.W2 -= (self.alpha) * np.matmul(delta2, np.transpose(a1))
        self.b2 -= self.alpha * delta2

        delta1 = np.matmul(np.transpose(self.W2), delta2) * sigmoid_derivative(z2)
        self.W1 -= self.alpha * np.matmul(np.transpose(self.input), delta1)
        self.b1 -= self.alpha * delta1


if __name__ == '__main__':
    BP = BPnetwork()
    BP.__init__()

    for i in range(10000):
        BP.propogate()

        if i % 100 == 0:
            _, _, _, a2 = BP.feedfoward()
            print(a2)

