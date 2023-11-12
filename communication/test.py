import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
from torchvision import datasets,transforms
from torch.autograd import Variable
import sys

def plot_loss(n):
    writer = SummaryWriter('./tb_log')
    fig, axe = plt.subplots(1, 1, figsize=(8, 5))
    plt.ion()#
    font1 = {'family': 'DejaVu Sans', 'size': '15'}
    # 饼状图各个部分的标签、值、颜色
    labels = ['food', 'clothing', 'housing', 'transport']
    values = [0.35, 0.15, 0.2, 0.3]
    for i, v in enumerate(values):
        writer.add_scalar("loss", v, global_step=i)
    colors = ['#D2ACA3', '#EBDFDF', '#DE6B58', '#E1A084']
    # 突出显示
    explode = [0, 0.1, 0, 0]
    # 标题
    #axe.set_title("daily cost", fontdict=font1)
    # 画饼状图
    wedge, texts, pcts = axe.pie(values, labels=labels, colors=colors, startangle=45, autopct='%3.1f%%'
                                 , explode=explode)
    axe.axis('equal')
    # 图例
    axe.legend(wedge, labels, fontsize=10, title='event', loc=2)
    # 设置文本的属性

    plt.setp(texts, size=12)
    plt.setp(pcts, size=12)
    #writer.add_figure(tag='test',figure=fig)
    plt.show()

def tb_plot():
    # fake data

    x = torch.linspace(-5, 5, 200)
    x = Variable(x)
    x_np = x.data.numpy()

    y_relu = F.relu(x).data.numpy()
    y_sigmoid = F.sigmoid(x).data.numpy()
    y_tanh = F.tanh(x).data.numpy()
    y_softplus = F.softplus(x).data.numpy()
    # y_softmax = F.softmax(x)

    fig = plt.figure(1, figsize=(8, 6))
    plt.subplot(221)
    plt.plot(x_np, y_relu, c='red', label='sigmoid')
    plt.ylim((-1, 5))
    plt.legend(loc='best')

    plt.subplot(222)
    plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
    plt.ylim((-0.2, 1.2))
    plt.legend(loc='best')

    plt.subplot(223)
    plt.plot(x_np, y_tanh, c='red', label='tanh')
    plt.ylim((-1.2, 1.2))
    plt.legend(loc='best')

    plt.subplot(224)
    plt.plot(x_np, y_softplus, c='red', label='softplus')
    plt.ylim((-0.2, 6))
    plt.legend(loc='best')

    writer = SummaryWriter('./tb_plot')
    #writer.add_figure(tag='activation_function', figure=fig)
    for i in range(200):

        writer.add_scalar('activation_function', i+1,i*2)
    #writer.add_graph(tag='activation_function', y_softplus)
    writer.close()

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(  # 输入torch.Size([64, 1, 28, 28])
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            # 用于搭建卷积神经网络的卷积层，主要的输入参数有输入通道数、
            # 输出通道数、卷积核大小、卷积核移动步长和Padding值。
            # 输出维度 = 1+(输入维度-卷积核大小+2*padding)/卷积核步长
            # 输出torch.Size([64, 64, 28, 28])

            torch.nn.ReLU(),  # 输出torch.Size([64, 64, 28, 28])
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # 输出torch.Size([64, 128, 28, 28])

            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2)
            # 主要的输入参数是池化窗口大小、池化窗口移动步长和Padding值
            # 输出torch.Size([64, 128, 14, 14])
        )

        self.dense = torch.nn.Sequential(  # 输入torch.Size([64, 14*14*128])
            torch.nn.Linear(14 * 14 * 128, 1024),
            # class torch.nn.Linear(in_features，out_features，bias = True)
            # 输出torch.Size([64, 1024])
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            # torch.nn.Dropout类用于防止卷积神经网络在训练的过程中
            torch.nn.Linear(1024, 10)
            # 输出torch.Size([64, 10])
        )
    def forward(self, x):  # torch.Size([64, 1, 28, 28])
        x = self.conv1(x)  # 输出torch.Size([64, 128, 14, 14])
        x = x.view(-1, 14 * 14 * 128)
        # view()函数作用是将一个多行的Tensor,拼接成一行，torch.Size([64, 14*14*128])
        x = self.dense(x)  # 输出torch.Size([64, 10])
        return x


if __name__ == "__main__":
    #plot_loss(20)
   # tb_plot()
    a = 6610344
    print(sys.getsizeof(a))
    #test7()
    """
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5],
                                                         std=[0.5])])
    data_train = datasets.MNIST(root="./data/",
                                transform=transform,
                                train=True,
                                download=True)
    data_loader_train = torch.utils.data.DataLoader(
        dataset=data_train,
        batch_size=64,
        shuffle=True)
    # images, labels = next(iter(data_loader_train))#迭代器
    # torch.Size([64, 1, 28, 28])
    images = torch.randn(64, 1, 28, 28)

    model = Model()

    writer = SummaryWriter('./tb_plot')
    for i in range(5):
        images = torch.randn(64, 1, 28, 28)
        writer.add_graph(model, input_to_model=images, verbose=False)

    writer.flush()
    writer.close()
    """

