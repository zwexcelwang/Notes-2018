#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/19 16:23
# @Author : zui
# @File : rn_sample1.py

import torch
import numpy as np
import torchvision
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from torch.autograd import Variable

# perpare data set
# train data
train_data = torchvision.datasets.ImageFolder('./data/train', transform=transforms.Compose(
    [
        transforms.Resize(500),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]))
print(len(train_data))
train_loader = DataLoader(train_data, batch_size=15, shuffle=True)

# test data
test_data = torchvision.datasets.ImageFolder('./data/test', transform=transforms.Compose(
    [
        transforms.Resize(500),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]))
print(len(test_data))
test_loader = DataLoader(test_data, batch_size=5, shuffle=True)

# prepare model
mode1_ft_res18 = torchvision.models.resnet18(pretrained=True)
# pretrained 设置为 True，会自动下载模型 所对应权重，并加载到模型中
for param in mode1_ft_res18.parameters():
    param.requires_grad = False
# 假设我们的分类任务只需要分 4类，那么我们应该做的是
# 1. 查看 resnet 的源码
# 2. 看最后一层的 名字是啥 （在 resnet 里是 self.fc = nn.Linear(512 * block.expansion, num_classes)）
# 3. 在外面替换掉这个层

num_fc = mode1_ft_res18.fc.in_features

# 只定义了一个全连接层，4是类别个数
# 修改后的模型除了输出层的参数是 随机初始化的，其他层都是用预训练的参数初始化的
mode1_ft_res18.fc = torch.nn.Linear(num_fc, 4)

'''
# 如果只想训练 最后一层的话，应该做的是：
# 1. 将其它层的参数 requires_grad 设置为 False
# 2. 构建一个 optimizer， optimizer 管理的参数只有最后一层的参数
# 3. 然后 backward， step 就可以了

# 这一步可以节省大量的时间，因为多数的参数不需要计算梯度
for para in list(resnet_model.parameters())[:-1]:
    para.requires_grad=False 

optimizer = optim.SGD(params=[resnet_model.fc.weight, resnet_model.fc.bias], l

'''

# loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
# parameters only train the last fc layer
optimizer = torch.optim.Adam(mode1_ft_res18.fc.parameters(), lr=0.001)

# start train
# label  not  one-hot encoder
EPOCH = 10
for epoch in range(EPOCH):
    train_loss = 0.
    train_acc = 0.
    for step, data in enumerate(train_loader):
        batch_x, batch_y = data
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        # batch_y not one hot
        # out is the probability of eatch class
        # such as one sample[-1.1009  0.1411  0.0320],need to calculate the max index
        # out shape is batch_size * class
        out = mode1_ft_res18(batch_x)
        loss = criterion(out, batch_y)
        train_loss += loss.item()
        # pred is the expect class
        # batch_y is the true label
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 14 == 0:
            print('Epoch: ', epoch, 'Step', step,
                  'Train_loss: ', train_loss / ((step + 1) * 20), 'Train acc: ', train_acc / ((step + 1) * 20))

    # print('Epoch: ', epoch, 'Train_loss: ', train_loss / len(train_data), 'Train acc: ', train_acc / len(train_data))

# test model
mode1_ft_res18.eval()
eval_loss = 0
eval_acc = 0
for step, data in enumerate(test_loader):
    batch_x, batch_y = data
    batch_x, batch_y = Variable(batch_x), Variable(batch_y)
    out = mode1_ft_res18(batch_x)
    loss = criterion(out, batch_y)
    eval_loss += loss.item()
    # pred is the expect class
    # batch_y is the true label
    pred = torch.max(out, 1)[1]
    test_correct = (pred == batch_y).sum()
    eval_acc += test_correct.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print('Test_loss: ', eval_loss / len(test_data), 'Test acc: ', eval_acc / len(test_data))