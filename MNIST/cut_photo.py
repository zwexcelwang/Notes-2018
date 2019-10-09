#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/9/30 16:07
# @Author : zui
# @File : cut_photo.py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/9/25 12:54
# @Author : zui
# @File : create_training.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random

DATADIR = r"C:\Workspaces\pycharm_workspace\MNIST\training-images"

CATEGORIES = ["1", "2"]

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # 连接两个或更多路径组件
        class_num = CATEGORIES.index(category)  # Dog表示为0，Cat表示为1

        save_path = r"C:\Workspaces\pycharm_workspace\MNIST\training"
        # print(class_num)

        for img in tqdm(os.listdir(path)):
            try:
                IMG_SIZE = 500
                # cv2.imread()：读入图片，共两个参数，
                # 第一个参数为要读入的图片文件名，
                # 第二个参数为如何读取图片，
                # 包括cv2.IMREAD_COLOR：读入一副彩色图片；
                # cv2.IMREAD_GRAYSCALE：以灰度模式读入图片；
                # cv2.IMREAD_UNCHANGED：读入一幅图片，并包括其alpha通道。

                image = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                # cv2.imshow('image', image)
                resize_image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

                cv2.save(os.path.join(save_path, category), img)

                training_data.append([resize_image, class_num])    # 把图片数组和分类标签加入数据集
            except Exception as e:
                # print('corrupt img', os.path.join(path,img))  # 报出错图片的路径
                pass


create_training_data()
random.shuffle(training_data)
print(len(training_data))


