##  DenseNet 

### 简介

**Paper：** Densely Connected Convolutional Networks 

### 参考

>  https://blog.csdn.net/sigai_csdn/article/details/82115254 
>
>  https://blog.csdn.net/MOU_IT/article/details/84938465 
>
> https://www.imooc.com/article/36508

### 介绍

**Inception网络主要是从网络的宽度方面改进网络的结构从而提高网络的表达能力，而ResNet主要是从网络的深度方面改进网络的结构来提高表达能力** 



ResNet模型的核心是通过建立前面层与后面层之间的“短路连接”（shortcuts，skip connection），这有助于训练过程中梯度的反向传播，从而能训练出更深的CNN网络。今天我们要介绍的是DenseNet模型，它的基本思路与ResNet一致，但是它建立的是前面所有层与后面层的密集连接（dense connection），它的名称也是由此而来。

 DenseNet**让网络的每一层的输入变成所有前面层的叠加（concat）**，然后把它的特征图传递给所有接下来的网络层。**（特征重用）**传统的CNN如果有L层则有L个连接，而DenseNet如果有**L层**，则有**1/2 L(L+1)**个连接，如下图所示： 



DenseNet作为另一种拥有较深层数的卷积神经网络,具有如下优点:

(1) 相比ResNet拥有更少的参数数量.

(2) 旁路加强了特征的重用.

(3) 网络更易于训练,并具有一定的正则效果.

(4) 缓解了gradient vanishing和model degradation的问题



DenseNet在提出时也做过假设:与其多次学习冗余的特征,特征复用是一种更好的特征提取方式.

 