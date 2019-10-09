#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/9/9 10:06
# @Author : zui
# @File : mnist.py

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data  # 这是TensorFlow 为了教学Mnist而提前设计好的程序

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # TensorFlow 会检测数据是否存在。当数据不存在时，系统会自动，在当前代码py文件位置，自动创建MNIST_data文件夹，并将数据下载到该件夹内。当执行完语句后，读者可以自行前往MNIST_data／文件夹下查看上述4 个文件是否已经被正确地下载
# 若因网络问题无法正常下载，可以前往MNIST官网http://yann.lecun.com/exdb/mnist/使用下载工具下载上述4 个文件， 并将它们复制到MNIST_data／文件夹中。


# 查看训练数据的大小
print(mnist.train.images.shape)  # (55000, 784)
print(mnist.train.labels.shape)  # (55000, 10)

# 查看验证数据的大小
print(mnist.validation.images.shape)  # (5000, 784)
print(mnist.validation.labels.shape)  # (5000, 10)

# 查看测试数据的大小
print(mnist.test.images.shape)  # (10000, 784)
print(mnist.test.labels.shape)  # (10000, 10)

print(mnist.train.images[0, :])  # 打印出第0张训练图片对应的向量表示
