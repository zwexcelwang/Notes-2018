## PyTorch入门

### 1. pytorch张量

张量只是多维数组。PyTorch中的张量类似于numpy的n

darrays。

定义一个简单的矩阵：

```python
import torch

a = torch.FloatTensor([2])
b = torch.FloatTensor([3])
print(a+b)


matrix = torch.randn(3, 3)
print(matrix)
print(matrix.t())
```

### 2. torch.nn.Conv2d

##### (*in_channels*, *out_channels*, *kernel_size*, *stride=1*, *padding=0*, *dilation=1*, *groups=1*, *bias=True*, *padding_mode='zeros'*)

input： (N, C~in~, H, W)

output：（N, C~out~, H~out~, W~out~）

- N是batch的大小
- C是通道数量
- H是输入的高度
- W是输入的宽度

N和C-in、C-out是人为指定，H，W是原始输入，H-out，W-out是通过公式计算出来的。

在卷积神经网络的卷积层之后总会添加**BatchNorm2**进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定