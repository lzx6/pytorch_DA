# -*- coding: utf-8 -*-
"""
@ project: WDGRL
@ author: lzx
@ file: no_transfer.py
@ time: 2019/6/17 14:00
"""
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_blobs
import torch.utils.data as Data

xs, ys = make_blobs(1000, centers=[[0, 0], [0, 10]], cluster_std=1.5)
xt, yt = make_blobs(1000, centers=[[50, -20], [50, -10]], cluster_std=1.5)
def plot_data(xs, ys, xt, yt):
    ys_pos_index = np.where(ys == 1)[0]
    ys_neg_index = np.where(ys == 0)[0]
    xs_pos = xs[ys_pos_index]
    xs_neg = xs[ys_neg_index]
    yt_pos_index = np.where(yt == 1)[0]
    yt_neg_index = np.where(yt == 0)[0]
    xt_pos = xt[yt_pos_index]
    xt_neg = xt[yt_neg_index]
    plt.scatter(xs_pos[:, 0], xs_pos[:, 1], c='r', s=8, alpha=0.7, label='source positive')
    plt.scatter(xs_neg[:, 0], xs_neg[:, 1], c='b', s=8, alpha=0.7, label='source negative')
    plt.scatter(xt_pos[:, 0], xt_pos[:, 1], c='r',marker='*', s=8, alpha=0.7, label='target positive')
    plt.scatter(xt_neg[:, 0], xt_neg[:, 1], c='b',marker='*', s=8, alpha=0.7, label='target negative')
    plt.legend()
    plt.grid(True)
    plt.title('toy dataset visualiz')
    plt.show()
# plot_data(xs,ys,xt,yt)
xs = torch.FloatTensor(xs)
xt = torch.FloatTensor(xt)
ys = torch.LongTensor(ys)
yt = torch.LongTensor(yt)
src_dataset = Data.TensorDataset(xs, ys)
tgt_dataset = Data.TensorDataset(xt,yt)
src_loader = Data.DataLoader(src_dataset,batch_size=32,shuffle=True)
tgt_loader = Data.DataLoader(tgt_dataset,batch_size=32,shuffle=True)
weight_clay = 1e-5
lr = 1e-3
training_epochs = 1000
def test(model,dataset_loader,every_epoch):
    model.eval()
    test_loss = 0
    corrcet = 0
    for tgt_data,tgt_label in dataset_loader:
        if torch.cuda.is_available():
            tgt_data = tgt_data.cuda()
            tgt_label = tgt_label.cuda()

        tgt_out = model(tgt_data)
        test_loss = criterion(tgt_out,tgt_label).item()
        pred = tgt_out.data.max(1,keepdim=True)[1]
        corrcet += pred.eq(tgt_label.data.view_as(pred)).cpu().sum()
    test_loss /= len(dataset_loader)
    print("epoch:{};average_loss:{};correct:{};total:{};accuracy:{}".format(every_epoch,test_loss,corrcet,
                                                                            len(dataset_loader.dataset),100.*float(corrcet)/len(dataset_loader.dataset)))
    return {
        'epoch': every_epoch,
        'average_loss': test_loss,
        'correct': corrcet,
        'total': len(dataset_loader.dataset),
        'accuracy': 100. * float(corrcet) / len(dataset_loader.dataset)
    }
class Net(nn.Module):
    def __init__(self,input_dim,hidden_dim,out_dim):
        super(Net,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,out_dim)

        self.fc1.weight.data.normal_(0,np.sqrt(input_dim/2))
        self.fc2.weight.data.normal_(0,np.sqrt(hidden_dim/2))
        self.relu = nn.ReLU(inplace=True)

    def forward(self ,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

net = Net(2,20,2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=net.parameters(),lr=lr,weight_decay=weight_clay)
for i in range(training_epochs):
    for index,(data,label) in enumerate(src_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
            net = net.cuda()
        optimizer.zero_grad()
        out = net(data)
        loss = criterion(out,label)
        loss.backward()
        optimizer.step()
    test(net,src_loader,i)
    test(net, tgt_loader, i)

