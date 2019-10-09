import os
import numpy as np
import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms as transforms
from torch.autograd import Variable

data_transform = transforms.Compose([
    transforms.Resize(500),
    transforms.ToTensor(),   # 转换成Tensor形式，并且数值归一化到[0.0, 1.0]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) #均值，方差都为0.5
    ])

# 用来读取数据集中所有图片的路径和标签
def get_file(file_dir):
    images = []  # 存放的是每一张图片对应的地址和名称
    temp = []    # 存放的是文件夹地址
    for root, sub_folders, files in os.walk(file_dir):
        # image directories
        for name in files:
            images.append(os.path.join(root, name))
            #print(images)
        #print(files)
        #print(images)

        # 读取当前目录选的文件夹
        for name in sub_folders:
            temp.append(os.path.join(root, name))
            #print(temp)
    #     print(sub_folders)
    #     print('a--------------a')
    # print(images)
    # print(temp)
    # print('--------------finish---------------')
    # 为数据集打标签，cat为0，dog为1
    labels = []
    error = 0
    for one_folder in temp:
        n_img = len(os.listdir(one_folder))  # os.listdir()返回指定的文件夹包含的文件或文件夹的名字的列表,再用len求该文件夹里面图像的数目
        # print(one_folder)
        # print(n_img)
        letter = one_folder.split('\\')[-1]  # 对路径进行切片,[-1]为取切片后的最后一个元素(也就是文件夹名称)。用于根据名称去判断数据集的样本类型
        #print(letter)
        if letter == '1':
            labels = np.append(labels, n_img * [0])  # 向labels里面添加1*n_img个0
        elif letter == '2':
            labels = np.append(labels, n_img * [1])
        elif letter == '3':
            labels = np.append(labels,n_img * [2])
        elif letter =='4':
            labels = np.append(labels,n_img * [3])
        else:
            error = error + 1
        # print(labels)
        # print(error)

    temp = np.array([images, labels])
    #print(temp)
    temp = temp.transpose()  # 矩阵转置
    #print(temp)
    np.random.shuffle(temp)  # shuffle() 是将序列的所有元素随机排序。
    #print(temp)

    image_list = list(temp[:, 0])  # 所有行，第0列
    label_list = list(temp[:, 1])  # 所有行，第1列
    # print(label_list)  # ['1.0', '1.0', '1.0', '0.0', '0.0', '0.0']

    label_list = [int(float(i)) for i in label_list]  # 把字符型标签转化为整型
    # print(label_list)  # [1, 1, 1, 0, 0, 0]
    print(len(image_list))
    print(len(label_list))
    return image_list, label_list



train_images_list, train_labels_list = get_file('training')
# print('image transform finished', images_list)
# print('label transform finished', labels_list)

test_images_list, test_labels_list = get_file('testing')
# print('image transform finished', images_list)
# print('label transform finished', labels_list)


# 以torch.utils.data.Dataset为基类创建MyDataset
class MyDataset(data.Dataset):
    # stpe1:初始化
    def __init__(self, image, label, transform=None, target_transform=None):
        imgs = []
        for i in range(len(image)):  # 遍历标签文件每行
            imgs.append((image[i], label[i]))  # 把图片名words[0]，标签int(words[1])放到imgs里
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  # 检索函数
        fn, label = self.imgs[index]  # 读取文件名、标签
        # img = Image.open(fn).convert('RGB')  # 通过PIL.Image读取图片
        img = Image.open(fn)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


train_data = MyDataset(image=train_images_list, label=train_labels_list, transform=data_transform)
test_data = MyDataset(image=test_images_list, label=test_labels_list, transform=data_transform)

train_loader = data.DataLoader(dataset=train_data, batch_size=15, shuffle=True)
test_loader = data.DataLoader(dataset=test_data, batch_size=5)


class my_cnn(torch.nn.Module):

    def __init__(self):
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        # 等价于 nn.Module.__init__()
        super(my_cnn, self).__init__()
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

        self.conv1 = torch.nn.Sequential(  # input shape (3, 500, 500)
            torch.nn.Conv2d(in_channels=3,  # input height
                            out_channels=16,  # n_filter
                            kernel_size=11,  # filter size
                            stride=3,  # filter step    (N-F+2*P)/S + 1
                            padding=0  # 填白, 想要con2d卷积出来的图片尺寸没有变化, stride=1, padding=(kernel_size-1)/2
                            ),  # output shape (16, 164, 164)
            # BatchNorm2d的参数num_features：一般输入参数为batch_size*num_features*height*width，即为其中特征的数量
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),

        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 164, 164)
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)   # (32, 82, 82)
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 3, 1),  # output shape (64, 28, 28)
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, 3, 1),  # output shape (128, 10, 10)
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)  # (128, 5, 5)
        )

        # 全连接层
        self.mlp1 = torch.nn.Linear(5 * 5 * 128, 500)    # in_features=4*256, out_features=100
        self.mlp2 = torch.nn.Linear(500, 100)    # in_features=100, out_features=10
        self.mlp3 = torch.nn.Linear(100, 4)
    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        x = self.mlp3(x)
        return x

cnn = my_cnn()

print(cnn)

# optimizer优化器
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

# loss_fun损失函数
loss_func = torch.nn.CrossEntropyLoss()

# training loop
for epoch in range(3):  # 训练所有整套数据2次
    for step,(x, y) in enumerate(train_loader):
        batch_x = Variable(x)   # torch.Size([128, 1, 28, 28]) # 128是batch_size
        batch_y = Variable(y)   # torch.Size([128])
        # 输入训练数据，获取最后输出
        output = cnn(batch_x)   # torch.Size([128,10])
        # 获取损失
        loss = loss_func(output, batch_y)
        # 使用优化器优化损失
        # 清空上一步梯度，更新参数值
        optimizer.zero_grad()
        # 误差反向传播，计算参数更新值
        loss.backward()
        # 优化器将参数更新值施加到net的parmeters上
        optimizer.step()
        if step % 100 == 0:
            for a,b in test_loader:
                test_x = Variable(a)
                test_y = Variable(b)
                out = cnn(test_x)
                # print('test_out:\t',torch.max(out,1)[1])
                # print('test_y:\t',test_y)
                print('Epoch：', epoch, 'step：', step, 'loss：', loss)
                prediction = torch.max(out, 1)[1]
                pred_y = prediction.numpy()
                test_y = test_y.numpy()
                accuracy = pred_y == test_y
                print('accuracy:\t', accuracy.mean())
                break


# import cv2
# import numpy as np
# import os
# # 建立数据集
# d1_path = "./training-images/1/"
# d2_path = "./training-images/2/"
# d3_path = "./training-images/3/"
# d4_path = "./training-images/4/"
# d1_imgs = [x for x in sorted(os.listdir(d1_path)) if x[-4:] == '.jpg']
# d2_imgs = [x for x in sorted(os.listdir(d2_path)) if x[-4:] == '.jpg']
# d3_imgs = [x for x in sorted(os.listdir(d3_path)) if x[-4:] == '.jpg']
# d4_imgs = [x for x in sorted(os.listdir(d4_path)) if x[-4:] == '.jpg']
#
# d1_data = np.empty((len(d1_imgs), 160, 60, 3), dtype='float32')
# d2_data = np.empty((len(d2_imgs), 160, 60, 3), dtype='float32')
# d3_data = np.empty((len(d3_imgs), 160, 60, 3), dtype='float32')
# d4_data = np.empty((len(d4_imgs), 160, 60, 3), dtype='float32')
#
#
# def dataloder():
# 	# 将所有图片变成（60，160）
#     for i, name in enumerate(d1_imgs):
#         im = cv2.imread(d1_path + name)
#         im=cv2.resize(im,dsize=(60,160))
#         d1_data[i] = im
#
#     for i, name in enumerate(d2_imgs):
#         im = cv2.imread(d2_path + name)
#         im=cv2.resize(im,dsize=(60,160))
#         d2_data[i] = im
#
#     for i, name in enumerate(d3_imgs):
#         im = cv2.imread(d3_path + name)
#         im=cv2.resize(im,dsize=(60,160))
#         d3_data[i] = im
#
#     for i, name in enumerate(d4_imgs):
#         im = cv2.imread(d4_path + name)
#         im=cv2.resize(im,dsize=(60,160))
#         d4_data[i] = im
#     labels = np.tile([1, 0], (len(d1_imgs), 1))
#     print(labels)
#     return d1_data,d2_data,d3_data,d4_data,labels
#
# dataloder()