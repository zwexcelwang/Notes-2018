#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/14 15:04
# @Author : zui
# @File : bp2_test2.py

from sklearn.datasets import load_digits
import matplotlib.pylab as pl

digits = load_digits()
print(digits.data.shape)
pl.gray()
pl.matshow(digits.images[0])
pl.show()