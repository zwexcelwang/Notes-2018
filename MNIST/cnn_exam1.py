#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/9/19 15:16
# @Author : zui
# @File : cnn_exam1.py

import torch
from torch.utils import data # 获取迭代数据
from torch.autograd import Variable # 获取变量
import torchvision
from torchvision.datasets import mnist # 获取数据集
import matplotlib.pyplot as plt


# 数据集的预处理
data_tf = torchvision.transforms.Compose(
    [
        # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，(height, width, 厚度？？个数？？)
        # 转换成形状为[C,H,W]，取值范围是[0, 1.0]的torch.FloadTensor
        torchvision.transforms.ToTensor(),

        # 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化。
        # 即：Normalized_image=(image-mean)/std。
        # 若都为0.5即x = 2x - 1, 最后的数据会在[-1, 1]
        torchvision.transforms.Normalize([0.5], [0.5])
    ]
)

data_path = r'/MNIST_data/'

# 获取数据集
train_data = mnist.MNIST(data_path, train=True, transform=data_tf, download=False)
test_data = mnist.MNIST(data_path, train=False, transform=data_tf, download=False)

train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = data.DataLoader(test_data, batch_size=100, shuffle=True)




# 定义网络结构
class CNNnet(torch.nn.Module):
    def __init__(self):
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        # 等价于 nn.Module.__init__()
        super(CNNnet, self).__init__()

        # nn.Conv2d返回的是一个Conv2d class的一个对象，该类中包含forward函数的实现
        # 当调用self.conv1(input)的时候，就会调用该类的forward函数
        '''

        in_channels：输入维度
        out_channels：输出维度
        kernel_size：卷积核大小
        stride：步长大小
        padding：补0
        dilation：kernel间距

        '''
        self.conv1 = torch.nn.Sequential(   # input shape (1, 28, 28)
            torch.nn.Conv2d(in_channels=1,  # input height
                            out_channels=16,#  n_filter
                            kernel_size=3,  # filter size
                            stride=2,       # filter step
                            padding=1       # 填白
                            ),              # output shape (16, 14, 14)
            # BatchNorm2d的参数num_features：一般输入参数为batch_size*num_features*height*width，即为其中特征的数量
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )


        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 2, 1),   # output shape (32, 7, 7)
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 2, 1),   # output shape (64, 4, 4)
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 2, 2, 0),   # output shape (64, 2, 2)
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(2*2*64, 100)    # in_features=256, out_features=100
        self.mlp2 = torch.nn.Linear(100, 10)        # in_features=100, out_features=10

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0),-1))
        x = self.mlp2(x)
        return x

model = CNNnet()


print(model)

# loss_fun
loss_func = torch.nn.CrossEntropyLoss()

# optimizer
opt = torch.optim.Adam(model.parameters(), lr=0.001)

loss_count = []

# training loop
for epoch in range(2): # 训练所有整套数据2次
    for i,(x,y) in enumerate(train_loader):
        batch_x = Variable(x) # torch.Size([128, 1, 28, 28]) # 128是batch_size
        batch_y = Variable(y) # torch.Size([128])
        # 输入训练数据，获取最后输出
        out = model(batch_x) # torch.Size([128,10])
        # 获取损失
        loss = loss_func(out, batch_y)
        # 使用优化器优化损失
        # 清空上一步梯度，更新参数值
        opt.zero_grad()
        # 误差反向传播，计算参数更新值
        loss.backward()
        # 优化器将参数更新值施加到net的parmeters上
        opt.step()

        # if i%20 == 0:
        #     loss_count.append(loss)
        #     print('{}:\t'.format(i), loss.item())
        #     torch.save(model, r'log_CNN.txt')
        if i % 100 == 0:
            for a,b in test_loader:
                test_x = Variable(a)
                test_y = Variable(b)
                out = model(test_x)
                # print('test_out:\t',torch.max(out,1)[1])
                # print('test_y:\t',test_y)
                print('Epoch：', epoch, 'i：', i, 'loss：', loss)
                prediction = torch.max(out, 1)[1]
                pred_y = prediction.numpy()
                test_y = test_y.numpy()
                accuracy = pred_y == test_y
                print('accuracy:\t', accuracy.mean())
                break

# # 取前2000个测试集样本
# test_x = Variable(torch.unsqueeze(test_loader, dim=1),
#          volatile=True).type(torch.FloatTensor)[:2000]/255
# # (2000, 28, 28) to (2000, 1, 28, 28), in range(0,1)
# test_y = test_data.test_labels[:2000]
#
# test_output = CNNnet(test_x[:10])
# pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
# print(pred_y, 'prediction number')
# print(test_y[:10].numpy(), 'real number')


# plt.figure('PyTorch_CNN_Loss')
# plt.plot(loss_count, label='Loss')
# plt.legend()
# plt.show()
