# -*- coding: utf-8 -*-
"""
@ project: WDGRL
@ author: lzx
@ file: model_3_channel.py
@ time: 2019/6/19 21:08
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradReverse(torch.autograd.Function):
    '''GRL'''
    @staticmethod###静态方式构造，无需实例化
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg()*ctx.constant
        return grad_output, None

    def grad_reverse(x ,constant):

        return GradReverse.apply(x,constant)

'''特征提取器'''
class Extractor(nn.Module):
    def __init__(self):
        super(Extractor,self).__init__()
        # self.conv1 = nn.Conv2d(3,64,kernel_size=5)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.conv2 = nn.Conv2d(64,50,kernel_size=5)
        # self.bn2 = nn.BatchNorm2d(50)
        self.conv1 = nn.Conv2d(3,32,kernel_size=5)
        self.conv2 = nn.Conv2d(32,48,kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

    def forward(self, x):
        x = x.expand(x.data.shape[0],3,28,28) #将1通道数据复制成3通道
        # x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)),2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.bn2(self.conv2(x))), 2))
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = x.view(-1, 48 * 4 * 4)
        return x

'''分类器'''
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.fc1 = nn.Linear(48*4*4,100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100,100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100,10)

    def forward(self,x):
        logits = F.relu(self.fc1(x))
        logits = self.fc2(F.dropout(logits))
        logits = F.relu(logits)
        logits = self.fc3(logits)

        return F.log_softmax(logits,1)###logsoftmax得出的结果都是负数

'''域分类器'''
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.fc1 = nn.Linear(48 * 4 * 4, 100)
        # self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self,x ,constant):
        x = GradReverse.grad_reverse(x,constant)
        logits = F.relu(self.fc1(x))
        logits = F.log_softmax(self.fc2(logits),1)
        return logits

def optimizer_scheduler(optimizer, p):
    """
    Adjust the learning rate of optimizer
    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75

    return optimizer
