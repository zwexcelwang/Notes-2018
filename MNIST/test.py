#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/9/19 13:35
# @Author : zui
# @File : test.py.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random


path = r"C:\Workspaces\pycharm_workspace\MNIST\test-images\1"

save_path = r"C:\Workspaces\pycharm_workspace\MNIST\testing\1"

for img in tqdm(os.listdir(path)):
    image = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
    print(image.shape)
    resize_image = cv2.resize(image, (500, 500))
    print(resize_image.shape)
    cv2.imwrite(os.path.join(save_path, img), resize_image)