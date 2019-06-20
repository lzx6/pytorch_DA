# -*- coding: utf-8 -*-
"""
@ project: WDGRL
@ author: lzx
@ file: no.py
@ time: 2019/6/17 20:19
"""
from data import Data_Mnist,Data_Mnist_M
from models import Classifier,Extractor,Discriminator,optimizer_scheduler
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from uitls import save
from torch.autograd import Variable

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

def train(f,c,data,optimizer,step):
    result = []
    src_data , src_label = data
    p = float(step)/total_steps
    if torch.cuda.is_available():
        src_data = src_data.cuda()
        src_label = src_label.cuda()
    optimizer = optimizer_scheduler(optimizer,p)
    optimizer.zero_grad()
    source_Z = f(src_data)
    class_pred = c(source_Z)
    class_loss = class_criterion(class_pred, src_label)
    loss = class_loss
    loss.backward()
    optimizer.step()
    result.append({
            'step': step,
            'total_steps': total_steps,
            'classification_loss': loss.item(),
        })
    if (step + 1) % 100 == 0:
        print('Train step:  [{:2d}/{:2d}]\t'
                      ' classification_loss: {:.6f}'.format(
                    step,
                    total_steps,
                    loss.item(),
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
        for index, data in enumerate(source_loader):
            p = float(index + start_steps) / total_steps
            res = train(feature_extrator, class_classifier, data, optimizer, index + start_steps)
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
    result_path = 'result_norm_no'
    import os
    os.makedirs(result_path, exist_ok=True)
    # torch.save(net.state_dict(), result_path + '/checkpoint.tar')
    save(training_sta, result_path + '/training_state.pkl')
    save(test_s_sta, result_path + '/test_s_sta.pkl')
    save(test_t_sta, result_path + '/test_t_sta.pkl')
# from data import get_usps,get_mnist
# from models import NET
# import torch
# import torch.nn as nn
# from uitls import save
# import numpy as np
# import random
# l2_param = 1e-5
# lr = 1e-4
# batch_size = 64
# num_steps = 10000
#
# def set_seed(seed):
#     if seed is None:
#         seed = random.randint(1, 10000)
#     print('seed:',seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#
# def train(model,optimizer,step):
#     result = []
#     src_data, src_label = iter_src.next()
#     # print(src_label.shape)
#     if src_label.dim() == 2:
#         src_label[torch.eq(src_label, 10)] = 0
#         src_label = torch.squeeze(src_label).long()
#     # # print(src_label)
#     # # print(labels)
#     # if src_label.dim() == 2:
#     #     src_label = src_label[:,0]
#     # print(src_data.shape,src_label)
#     if torch.cuda.is_available():
#         src_data = src_data.cuda()
#         src_label = src_label.cuda()
#     optimizer.zero_grad()
#     _,out = model(src_data)
#     loss_classifier = criterion(out, src_label)
#     loss = loss_classifier
#     loss.backward()
#     optimizer.step()
#     # optimizer.zero_grad()
#     result.append({
#         'step': step,
#         'total_steps': num_steps,
#         'classification_loss': loss_classifier.item(),
#     })
#     if (step+1) % 100 == 0:
#         print('Train step:  [{:2d}/{:2d}]\t'
#               ' classification_loss: {:.6f}'.format(
#             step,
#             num_steps,
#             loss_classifier.item(),
#         ))
#     return result
#
# def test(model,dataset_loader,every_epoch):
#
#     model.eval()
#     with torch.no_grad():
#         test_loss = 0
#         corrcet = 0
#         for tgt_data,tgt_label in dataset_loader:
#             if tgt_label.dim() == 2:
#                 tgt_label[torch.eq(tgt_label, 10)] = 0
#                 tgt_label = torch.squeeze(tgt_label).long()
#             if torch.cuda.is_available():
#                 tgt_data = tgt_data.cuda()
#
#                 tgt_label = tgt_label.cuda()
#
#             _,tgt_out= model(tgt_data)
#             test_loss += nn.CrossEntropyLoss()(tgt_out,tgt_label).item()
#             pred = tgt_out.data.max(1,keepdim=True)[1]
#             # print(pred)
#             # print(tgt_label)
#             corrcet += pred.eq(tgt_label.data.view_as(pred)).cpu().sum()
#             # print(corrcet)
#         test_loss /= len(dataset_loader)
#     # print(test_loss)
#     a = test_loss
#     return {
#         'epoch': every_epoch,
#         'average_loss': a,
#         'correct': corrcet,
#         'total': len(dataset_loader.dataset),
#         'accuracy': 100. * float(corrcet) / len(dataset_loader.dataset)
#     }
#
# if __name__ == '__main__':
#     mnist_path = 'F:/刘子绪/数据/数据image/mnist'
#     usps_path = 'F:/刘子绪/数据/数据image/uspmnist'
#
#     train_mnist_loader = get_mnist(root=mnist_path, batch_size=batch_size, train=True)
#     test_mnist_loader = get_mnist(root=mnist_path, batch_size=batch_size, train=False)
#     test_usp_loader = get_usps(root=usps_path, batch_size=batch_size, train=False)
#     print(len(train_mnist_loader.dataset), len(test_mnist_loader.dataset), len(test_usp_loader.dataset))
#     set_seed(None)
#     criterion = nn.CrossEntropyLoss()
#     net = NET(num_classes=10)
#     if torch.cuda.is_available():
#         net = net.cuda()
#         criterion = criterion.cuda()
#     optimizer = torch.optim.Adam([{'params': net.parameters()},
#                                 ],
#                                 lr=lr,weight_decay=l2_param)
#     training_sta = []
#     test_s_sta = []
#     test_t_sta = []
#     for step in range(num_steps):
#         if step % len(train_mnist_loader) == 0:
#             iter_src = iter(train_mnist_loader)
#         res = train(net,optimizer,step)
#         training_sta.append(res)
#         if (step+1) % 100 == 0:
#             test_source = test(net, test_mnist_loader, step)
#             test_target = test(net, test_usp_loader, step)
#             test_s_sta.append(test_source)
#             test_t_sta.append(test_target)
#             print('###Test Source: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
#                 step + 1,
#                 test_source['average_loss'],
#                 test_source['correct'],
#                 test_source['total'],
#                 test_source['accuracy'],
#             ))
#             print('###Test Target: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
#                 step + 1,
#                 test_target['average_loss'],
#                 test_target['correct'],
#                 test_target['total'],
#                 test_target['accuracy'],
#             ))
#     result_path = 'result_norm_no'
#     import os
#
#     os.makedirs(result_path, exist_ok=True)
#     torch.save(net.state_dict(), result_path + '/checkpoint.tar')
#     save(training_sta, result_path + '/training_state.pkl')
#     save(test_s_sta, result_path + '/test_s_sta.pkl')
#     save(test_t_sta, result_path + '/test_t_sta.pkl')