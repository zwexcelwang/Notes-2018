#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/14 15:13
# @Author : zui
# @File : bp2_test3.py

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from bp2 import NeuralNetwork
from sklearn.model_selection  import train_test_split

# 加载数据集
digits = load_digits()
X = digits.data
y = digits.target
# 处理数据，使得数据处于0,1之间，满足神经网络算法的要求
X -= X.min()
X /= X.max()

# 层数：
# 输出层10个数字
# 输入层64因为图片是8*8的，64像素
# 隐藏层假设100
nn = NeuralNetwork([64, 100, 10], 'logistic')
# 分隔训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 转化成sklearn需要的二维数据类型
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)
print("start fitting")
# 训练3000次
nn.fit(X_train, labels_train, epochs=3000)
predictions = []
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i])
    # np.argmax:第几个数对应最大概率值
    predictions.append(np.argmax(o))

# 打印预测相关信息
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
