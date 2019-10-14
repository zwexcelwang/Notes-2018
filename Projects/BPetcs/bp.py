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
    # 定义神经网络
    def __init__(self):
        self.alpha = 0.5    # 学习率
        '''
        0.15 0.20       input1(0.05)   =   output1(0.0275)    +   0.35  (sigmoid之后0.59326999)
        0.25 0.30   *   input2(0.10)   =   output2(0.0425)    +   0.35  (sigmoid之后0.59688438)
        
        0.40 0.45        0.59326999    =   output3(0.50590597)    +   0.90 (sigmoid之后0.8031194) (expect 0.05)
        0.50 0.55   *    0.59688438    =   output4(0.6249214)     +   0.60 (sigmoid之后0.77292847) (expect 0.90)
        '''
        self.W1 = np.array([[0.15, 0.20],
                            [0.25, 0.30]])   #输入层到卷积层的系数矩阵
        self.b1 = np.array([[0.35],
                            [0.35]])
        self.W2 = np.array([[0.40, 0.45],
                            [0.50, 0.55]])
        self.b2 = np.array([[0.90],
                            [0.60]])
        self.input = np.array([[0.05],
                               [0.10]])
        self.output = np.array([[0.05],
                                [0.9]])

    # 前向传播
    def feedfoward(self):
        Z1 = np.matmul(self.W1, self.input) + self.b1
        A1 = sigmoid(Z1)
        Z2 = np.matmul(self.W2, A1) + self.b2
        A2 = sigmoid(Z2)
        return Z1, A1, Z2, A2

    #反向传播
    def propogate(self):
        z1, a1, z2, a2 = self.feedfoward()

        '''
        损失函数C = 1/2 * (output - a2) ^ 2
        求导： -(output - a2)
        '''
        delta2 = -(self.output - a2) * sigmoid_derivative(z2)
        # np.transpose(a1) = [[0.59326999(x1) 0.59688438](x2)]
        # delta2 = [ [d1],
        #            [d2] ]
        # W2 = [ [d1*x1 d1*x2],
        #        [d2*x1 d2*x2] ]
        print('delta2', delta2)
        self.W2 -= (self.alpha) * np.matmul(delta2, np.transpose(a1))
        self.b2 -= self.alpha * delta2

        print('self.W2', self.W2)
        print('np.transpose(self.W2)', np.transpose(self.W2))

        '''
        若self.W2 = [[0.01191817 0.05984106]
                 [0.55688395 0.60719122]]
        则np.transpose(self.W2) = [[0.01191817 0.55688395]
                               [0.05984106 0.60719122]]
        '''

        delta1 = np.matmul(np.transpose(self.W2), delta2) * sigmoid_derivative(z2)

        # delta1 = [ []
        #            []]
        # np.transpose(self.input) = [[.. ..]]
        # np.matmul(np.transpose(self.input), delta1) = [[..]]

        print('delta1', delta1)
        print('np.transpose(self.input)', np.transpose(self.input))

        self.W1 -= self.alpha * np.matmul(np.transpose(self.input), delta1)

        # print('np.matmul(np.transpose(self.input), delta1)', np.matmul(np.transpose(self.input), delta1))
        # print('self.W1', self.W1)

        # W1 = [ [.. ..],
        #        [.. ..] ]
        self.b1 -= self.alpha * delta1


if __name__ == '__main__':
    BP = BPnetwork()
    BP.__init__()

    for i in range(10000):
        BP.propogate()

        if i % 100 == 0:
            _, _, _, a2 = BP.feedfoward()
            print(a2)

