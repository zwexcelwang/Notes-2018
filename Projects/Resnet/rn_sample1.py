#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/19 16:23
# @Author : zui
# @File : test.py

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
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]))
print(len(train_data))
train_loader = DataLoader(train_data, batch_size=20, shuffle=True)

# test data
test_data = torchvision.datasets.ImageFolder('./data/test', transform=transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]))
print(len(train_data))
test_loader = DataLoader(test_data, batch_size=20, shuffle=True)

# prepare model
mode1_ft_res18 = torchvision.models.resnet18(pretrained=True)
for param in mode1_ft_res18.parameters():
    param.requires_grad = False
num_fc = mode1_ft_res18.fc.in_features
mode1_ft_res18.fc = torch.nn.Linear(num_fc, 4)

# loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
# parameters only train the last fc layer
optimizer = torch.optim.Adam(mode1_ft_res18.fc.parameters(), lr=0.001)

# start train
# label  not  one-hot encoder
EPOCH = 100
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