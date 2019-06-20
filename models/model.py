# -*- coding: utf-8 -*-
"""
@ project: WDGRL
@ author: lzx
@ file: model.py
@ time: 2019/6/17 20:09
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x, constant):
        ctx.constant = constant
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_outputs):
        grad_outputs = grad_outputs.neg()*ctx.constant

        return grad_outputs,None

    def grad_reverse( x, constant):
        return GradReverse.apply(x,constant)

class NET(nn.Module):
    def __init__(self,num_classes):
        super(NET,self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*64,1024),
            nn.ReLU(),

        )
        self.classifer = nn.Linear(1024,num_classes)

    def forward(self,x):
        f = self.feature(x)
        f = f.view(x.shape[0],-1)
        f = self.fc(f)
        out = self.classifer(f)
        return f,out

class Discriminator(nn.Module):
    def __init__(self,hidden_dim):
        super(Discriminator,self).__init__()
        # self.re = nn.ReLU(),
        self.fc1 = nn.Linear(hidden_dim, 2)
        # self.bn1 = nn.BatchNorm1d(100)
        # self.fc2 = nn.Linear(100, 2)

    def forward(self,x ,constant):
        # x = self.re(x),
        x = GradReverse.grad_reverse(x,constant)
        logits = self.fc1(x)
        return logits