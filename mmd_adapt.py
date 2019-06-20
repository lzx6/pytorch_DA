# -*- coding: utf-8 -*-
"""
@ project: WDGRL
@ author: lzx
@ file: mmd_adapt.py
@ time: 2019/6/18 17:07
"""
from data import Data_Mnist,Data_Mnist_M
from models import Classifier,Extractor,Discriminator,optimizer_scheduler,mmd_rbf_
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from uitls import save


'''params'''
batch_size = 128
lr = 0.01
momentum = 0.9
total_epochs = 100
source_dataset_train, source_dataset_test = Data_Mnist()
target_dataset_train, target_dataser_test = Data_Mnist_M()

source_loader = torch.utils.data.DataLoader(source_dataset_train, batch_size = batch_size, shuffle = True)
target_loader = torch.utils.data.DataLoader(target_dataset_train, batch_size = batch_size, shuffle = True)
s_test_loader = torch.utils.data.DataLoader(source_dataset_test, batch_size = batch_size, shuffle = True)
t_test_loader = torch.utils.data.DataLoader(target_dataser_test, batch_size = batch_size, shuffle = True)
total_steps = total_epochs*len(source_loader)
'''定义网络框架'''
feature_extrator = Extractor()
class_classifier = Classifier()
class_criterion = nn.NLLLoss()
optimizer = optim.SGD([{'params': feature_extrator.parameters()},
                            {'params': class_classifier.parameters()}], lr= lr, momentum= momentum)

if torch.cuda.is_available():
    feature_extrator = feature_extrator.cuda()
    class_classifier = class_classifier.cuda()
    class_criterion = class_criterion.cuda()

def train(f,c,source,target,optimizer,step):
    result = []
    source_data, source_label = source
    target_data, target_label = target
    # torchvision.utils.save_image(source_data,'mnist.png')
    # torchvision.utils.save_image(target_data, 'mnist_M.png')
    size = min((source_data.shape[0], target_data.shape[0]))
    # print(size)
    source_data, source_label = source_data[0:size, :, :, :], source_label[0:size]
    target_data, target_label = target_data[0:size, :, :, :], target_label[0:size]
    p = float(step)/total_steps
    gamma = 2 / (1 + np.exp(-10 * p)) - 1
    if torch.cuda.is_available():
        src_data = source_data.cuda()
        src_label = source_label.cuda()
        tgt_data = target_data.cuda()
    optimizer = optimizer_scheduler(optimizer,p)
    optimizer.zero_grad()
    source_Z = f(src_data)
    target_Z = f(tgt_data)
    class_pred = c(source_Z)
    class_loss = class_criterion(class_pred, src_label)
    mmd_loss = mmd_rbf_(source_Z,target_Z,[1,5,10])
    loss = class_loss+gamma*mmd_loss
    loss.backward()
    optimizer.step()
    result.append({
            'step': step,
            'total_steps': total_steps,
            'classification_loss': class_loss.item(),
            'mmd loss': mmd_loss.item()
        })
    if (step + 1) % 100 == 0:
        print('Train step:  [{:2d}/{:2d}]\t'
                      ' classification_loss: {:.6f}   mmd_loss: {:.6f}'.format(
                    step,
                    total_steps,
                    class_loss.item(),
                    mmd_loss.item()
                ))
    return result

def test(f,c, dataset_loader, every_epoch):
    f.eval()
    c.eval()
    with torch.no_grad():
        test_loss = 0
        corrcet = 0
        for tgt_data,tgt_label in dataset_loader:

            if torch.cuda.is_available():
                tgt_data = tgt_data.cuda()

                tgt_label = tgt_label.cuda()

            tgt_out= f(tgt_data)
            tgt_out = c(tgt_out)
            test_loss += nn.NLLLoss()(tgt_out,tgt_label).item()
            pred = tgt_out.data.max(1,keepdim=True)[1]
            # print(pred)
            # print(tgt_label)
            corrcet += pred.eq(tgt_label.data.view_as(pred)).cpu().sum()
            # print(corrcet)
        test_loss /= len(dataset_loader)
    return {
        'epoch': every_epoch,
        'average_loss': test_loss,
        'correct': corrcet,
        'total': len(dataset_loader.dataset),
        'accuracy': 100. * float(corrcet) / len(dataset_loader.dataset)
    }

if __name__ == '__main__':
    training_sta = []
    test_s_sta = []
    test_t_sta = []
    for epoch in range(total_epochs):
        feature_extrator.train()
        class_classifier.train()
        start_steps = epoch * len(source_loader)
        for index, (source, target) in enumerate(zip(source_loader, target_loader)):
            p = float(index + start_steps) / total_steps
            res = train(feature_extrator, class_classifier, source,target, optimizer, index + start_steps)
            training_sta.append(res)

        test_source = test(feature_extrator,class_classifier, s_test_loader, epoch)
        test_target = test(feature_extrator, class_classifier, t_test_loader, epoch)

        test_s_sta.append(test_source)
        test_t_sta.append(test_target)
        print('###Test Source: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            epoch + 1,
            test_source['average_loss'],
            test_source['correct'],
            test_source['total'],
            test_source['accuracy'],
        ))
        print('###Test Target: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            epoch + 1,
            test_target['average_loss'],
            test_target['correct'],
            test_target['total'],
            test_target['accuracy'],
        ))
    result_path = 'result_norm_mmd'
    import os
    os.makedirs(result_path, exist_ok=True)
    # torch.save(net.state_dict(), result_path + '/checkpoint.tar')
    save(training_sta, result_path + '/training_state.pkl')
    save(test_s_sta, result_path + '/test_s_sta.pkl')
    save(test_t_sta, result_path + '/test_t_sta.pkl')