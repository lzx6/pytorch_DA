# -*- coding: utf-8 -*-
"""
@ project: WDGRL
@ author: lzx
@ file: mnist_usps.py
@ time: 2019/6/17 19:43
"""
import gzip
import os
import pickle
import urllib
import torch
from torchvision import datasets, transforms
import torch.utils.data as data
# import os
import numpy as np

__all__ = ['USPS','get_mnist','get_usps']

class USPS(data.Dataset):
    """USPS Dataset.

    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"

    def __init__(self, root, train=True, transform=None, download=False):
        """Init USPS dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.filename = "usps_28x28.pkl"
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None

        # download dataset.
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        self.train_data, self.train_labels = self.load_samples()
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], ::]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]
        self.train_data *= 255.0
        self.train_data = self.train_data.transpose(
            (0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """Get images and target for data loader.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.train_data[index, ::], self.train_labels[index]
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([np.int64(label)])
        # label = torch.FloatTensor([label.item()])
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        """Download dataset."""
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, os.path.abspath(filename)))
        urllib.request.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    def load_samples(self):
        """Load sample images from dataset."""
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels
def get_mnist(root,train,batch_size):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=(0.5,0.5,0.5),
                                        std = (0.5,0.5,0.5),
                                    )])
    dataset = datasets.MNIST(root = root,
                             train = train,
                             transform = transform)
    dataloader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = True
    )
    return dataloader

def get_usps(root,train,batch_size):
    transform = transforms.Compose([transforms.ToTensor(),
                                    # transforms.Normalize(
                                    #     mean=(0.5, 0.5, 0.5),
                                    #     std=(0.5, 0.5, 0.5),
                                    # )
                                    ])
    dataset = USPS(root=root,
                             train=train,
                             transform=transform)
    # images,labels  = dataset.load_samples()
    # print(images.shape,labels.shape)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return dataloader

# mnist_path = 'F:/刘子绪/数据/数据image/mnist'
usps_path = 'F:/刘子绪/数据/数据image/uspmnist'
# dataloader = get_mnist(mnist_path,batch_size=100,train=True)
# print(len(dataloader.dataset))
dataloader = get_usps(usps_path,batch_size=100,train=False)
# print(dataloader[])
# print(len(dataloader.dataset))
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# print(input.shape,target.shape)