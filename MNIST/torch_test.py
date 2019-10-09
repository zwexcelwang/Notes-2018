#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/9/19 14:52
# @Author : zui
# @File : torch_test.py

import torch
from torch.autograd import Variable
import numpy as np
import torchvision

data = np.random.randint(0, 255, size=300)
img = data.reshape(10, 10, 3) # 10*10的三个，也就是rgb

img_tensor = torchvision.transforms.ToTensor()(img) # 转换成tensor
print(img_tensor)
img_tensor_normalize = torchvision.transforms.Normalize([0.5], [0.5])(img_tensor)
print(img_tensor_normalize)

'''

matrix = torch.randn(3, 3)
print(matrix)
print(matrix.t())


a = torch.FloatTensor([2])
b = torch.FloatTensor([3])
print(a+b)

'''