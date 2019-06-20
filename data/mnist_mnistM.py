# -*- coding: utf-8 -*-
"""
@ project: WDGRL
@ author: lzx
@ file: mnist_mnistM.py
@ time: 2019/6/19 21:10
"""
import torch.utils.data as data
from PIL import Image
import os
from torchvision import transforms
from torchvision import datasets

img_transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
'''读取mnist_M文件'''
class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list='F:\刘子绪\数据\mnist_m', transform=None):
        self.root = data_root
        self.transform = transform
        '''文件字符串'''
        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])###data[:-3]代表图像命名 # 00000000.png 5
            self.img_labels.append(data[-2])###data[-2]表示图像标签

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data

'''生成数据集MNIST_M'''
def Data_Mnist_M():
    '''返回Mnist_M的训练数据和测试数据'''
    data_root = 'F:\刘子绪\数据\mnist_m'
    data_root_train = os.path.join(data_root,'mnist_m_train')
    label_train = os.path.join(data_root,'mnist_m_train_labels.txt')
    dataset_target_train = GetLoader(data_root=data_root_train,data_list=label_train,transform=img_transform)

    data_root_test = os.path.join(data_root,'mnist_m_test')
    label_test = os.path.join(data_root,'mnist_m_test_labels.txt')
    dataset_target_test =GetLoader(data_root=data_root_test,data_list=label_test,transform=img_transform)

    return dataset_target_train, dataset_target_test

def Data_Mnist():
    dataset_source_train = datasets.MNIST(
        root='F:\刘子绪\数据\数据image\mnist',
        train=True,
        transform=img_transform,
    )
    dataset_source_test = datasets.MNIST(
        root='F:\刘子绪\数据\数据image\mnist',
        train=False,
        transform=img_transform,
    )
    return dataset_source_train , dataset_source_test

# source_dataset_train, source_dataset_test = Data_Mnist()
# target_dataset_train, target_dataser_test = Data_Mnist_M()
# train,test= Data_Mnist_M()
# # train, test = Data_Mnist()
# print(len(train),len(test))


