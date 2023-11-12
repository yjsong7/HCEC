import matplotlib.pyplot as plt
from PIL import Image
import struct
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os  # 目的是使ToTensor正确运行

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class FashionMNIST_IMG(Dataset):
    """
    自定义FMNIST数据集读取，并使用DataLoader加载器加载数据
    """

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(FashionMNIST_IMG, self).__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train:  # train sets
            images_file = root + r'/train-images.idx3-ubyte'
            labels_file = root + r'/train-labels.idx1-ubyte'
        else:
            images_file = root + r'/t10k-images.idx3-ubyte'
            labels_file = root + r'/t10k-labels.idx1-ubyte'

        # 读取二进制数据
        offset1, offset2 = 0, 0
        fp_img = open(images_file, 'rb').read()
        fp_label = open(labels_file, 'rb').read()

        # 解析文件头信息，依次为魔数、图片数量、每张图片的高、宽
        magics1, num_img, rows, cols = struct.unpack_from('>IIII', fp_img, offset1)
        magics2, num_label = struct.unpack_from('>II', fp_label, offset2)

        # 解析数据集
        offset1 += struct.calcsize('>IIII')
        offset2 += struct.calcsize('>II')
        # img_fmt = '>'+str(rows*cols)+'B'    #图像数据像素值的类型为unsignedchar型，对应的format格式为B
        # 这里的图像大小为28*28=784，为了读取784个B格式数据，如果没有则只会读取一个值
        # label_fmt = '>B'

        self.images = np.empty((num_img, rows, cols))
        self.labels = np.empty(num_label)

        assert num_img == num_label  # 判断图像个数是否等于标签个数，成立则往下执行

        for i in range(num_img):
            self.images[i] = np.array(struct.unpack_from('>' + str(rows * cols) + 'B', fp_img, offset1)).reshape(
                (rows, cols))
            self.labels[i] = struct.unpack_from('>B', fp_label, offset2)[0]
            offset1 += struct.calcsize('>' + str(rows * cols) + 'B')
            offset2 += struct.calcsize('>B')

    def __getitem__(self, item):
        img = self.images[item]
        label = self.labels[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)

    def get_label(self, n):
        """获得第n个数字对应的标签文本"""
        text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                       'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return text_labels[int(n)]

    def get_labels(self, labels):  # @save
        """返回Fashion-MNIST数据集的所有标签文本
        如labels = [1,3,5,3,6,2]
        此函数具有迭代器功能
        """
        text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                       'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [text_labels[int(i)] for i in labels]


if __name__ == '__main__':
    #path = r"/home/syj/project/federated-learning/data/FashionMNIST/raw"
    path = "/home/syj/project/federated-learning/data/FashionMNIST/raw"
    #path = path.replace('\\', '/')
    trans = transforms.ToTensor()  # 加载数据并将其转为张量，可以设置为None

    # 定义一个实例对象
    train_dataset = FashionMNIST_IMG(path, train=True, transform=trans)

    # =========方式二：通过迭代器读取小批量数据====================
    """
    在每次迭代中，数据加载器每次都会读取一小批量数据，大小为batch_size。 
    通过内置数据迭代器，我们可以随机打乱了所有样本，从而无偏见地读取小批量。
    batch_size=80  表示小批量个数为80个图像
    shuffle=True表示打乱顺序
    """
    train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
    for img, label in train_loader:
            print("图像和标签类型：", type(img), type(label), "图像和标签尺寸", img.shape, label.shape)
            img2 = np.array(np.uint8(img)).reshape(80, 28, 28)  # 转为三维矩阵
        # 或
        # for k, (img, label) in enumerate(train_loader):
        #     print("第", k + 1, "块图像和标签类型：", type(img), type(label), "图像和标签尺寸", img.shape, label.shape)