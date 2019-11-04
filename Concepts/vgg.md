## VGG

> @author：zui
>
> @time：2019-11-04
>
> @ps：仅供个人学习用

### 简介

**Paper**：Very Deep Convolutional Networks for Large-Scale Image Recognition

### 参考

#### paper翻译参考链接

> https://www.jianshu.com/p/9d6082068f53
>
> https://cloud.tencent.com/developer/article/1435343
>
> https://blog.csdn.net/u014114990/article/details/50715548
>
> https://blog.csdn.net/stdcoutzyx/article/details/39736509

#### vgg网络结构

> http://blog.csdn.net/amusi1994/article/details/81461968
>
> https://blog.csdn.net/marsjhao/article/details/72955935

### 原理

VGG16相比AlexNet的一个改进是采用连续的几个3x3的卷积核代替AlexNet中的较大卷积核（11x11，7x7，5x5）。对于**给定的感受野**（与输出有关的输入图片的局部大小），采用堆积的小卷积核是优于采用大的卷积核，**因为多层非线性层可以增加网络深度来保证学习更复杂的模式，而且代价还比较小**（参数更少）

简单来说，在VGG中，使用了3个3x3卷积核来代替7x7卷积核，使用了2个3x3卷积核来代替5*5卷积核，这样做的主要目的是在**保证具有相同感知野的条件下，提升了网络的深度，在一定程度上提升了神经网络的效果**。

VGGNet论文中全部使用了**3\*3的卷积核和2\*2的池化核**，通过不断加深网络结构来提升性能，证明 卷积神经网络的深度增加和小卷积核的使用对网络的最终分类识别效果有很大的作用。 

到目前为止，VGGNet依然经常被用来提取图像特征

**两个3\*3的卷积层串联相当于1个5\*5的卷积层**，即一个像素会跟周围5*5的像素产生关联，可以说感受野大小为5\*5。而**3个3\*3的卷积层串联的效果则相当于1个7\*7的卷积层**



