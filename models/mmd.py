# -*- coding: utf-8 -*-
"""
@ project: WDGRL
@ author: lzx
@ file: mmd.py
@ time: 2019/6/18 16:10
"""

import torch

min_var_test = 1e-8

src = torch.randn(64,100)
tgt = torch.randn(64,100)
def guassian_kernel(src,tgt,kernel_mul = 2,kernel_num = 5,fix_sigma = None):
    n_samples = int(src.size()[0])+int(tgt.size()[0])
    total = torch.cat([src,tgt],dim = 0)# 按列合并 (n_samples,feature_dim)
    total0 = total.unsqueeze(0).expand(int(total.size(0)),int(total.size(0)),int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)),int(total.size(0)),int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf_(src,tgt,sigma_list):
    batch_size = src.shape[0]
    kernels = guassian_kernel(src, tgt,
                              kernel_mul=2, kernel_num=5, fix_sigma=None)
    XX = kernels[:batch_size,:batch_size]
    YY = kernels[batch_size:,batch_size:]
    XY = kernels[:batch_size,batch_size:]
    YX = kernels[batch_size:,:batch_size]
    loss = torch.mean(XX+YY-XY-YX)
    return loss
#
# def rbf_kernel(X,Y,sigma_list):
#     '''
#     :param X:
#     :param Y:
#     :param sigma_list:
#     :return:
#     '''
#     assert(X.size(0)==Y.size(0))
#     m = X.size(0)
#     Z = torch.cat((X,Y),0) #
#     ZZT = torch.mm(Z,Z.t())
#     # print(ZZT.shape)
#     diag_ZZT = torch.diag(ZZT).unsqueeze(1)
#     Z_norm_2 = diag_ZZT.expand_as(ZZT)
#     exponent = Z_norm_2+Z_norm_2.t()-2*ZZT
#     K = 0.0
#     for sigma in sigma_list:
#         gamma = 1.0/(2*sigma**2+min_var_test)
#         K += torch.exp(-gamma*exponent)
#     return K[:m,:m],K[:m,m:],K[m:,m:],len(sigma_list)
#
# def mmd2(K_XX,K_XY,K_YY,biased = False):
#     m = K_XX.size(0)
#     diag_X = torch.diag(K_XX)  # (m,)
#     diag_Y = torch.diag(K_YY)  # (m,)
#     sum_diag_X = torch.sum(diag_X)
#     sum_diag_Y = torch.sum(diag_Y)
#
#     Kt_XX_sums = K_XX.sum(dim=1) - diag_X #减去对角，及数据本身与本身之间的关联，然后除以数目时对应的也少一个
#     Kt_YY_sums = K_YY.sum(dim=1) - diag_Y
#     K_XY_sums_0 = K_XY.sum(dim=0)
#
#     Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
#     Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
#     K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e
#
#     if biased:
#         mmd = ((Kt_XX_sum + sum_diag_X) / (m * m)
#                 + (Kt_YY_sum + sum_diag_Y) / (m * m)
#                 - 2.0 * K_XY_sum / (m * m))
#     else:
#         mmd = (Kt_XX_sum / (m * (m - 1))
#                 + Kt_YY_sum / (m * (m - 1))
#                 - 2.0 * K_XY_sum / (m * m))
#
#     return mmd
#
# def mix_rbf_mmd_loss(X,Y,sigma_list,biased=True):
#     K_XX, K_XY, K_YY, d = rbf_kernel(X, Y, sigma_list)
#     return mmd2(K_XX, K_XY, K_YY,  biased=biased)

# torch.manual_seed(1)
# x = torch.randn(5,4)
# y = torch.randn(5,4)
# mmd_loss = mix_rbf_mmd_loss(x,y,[1,2,10])
# print(mmd_loss)



import numpy as np
# x = np.array([[1,2],[3,4],[5,6]])
# y = np.array([1,2,3])
# print(y)
# # y = np.array([[1],[2]])
# print(x*y)
# print(x.shape,y.shape)
# y=np.array([1,2])
# x=np.array([[1,2,3], [3,4,5]])
# print(y.shape)
# print(x.shape)
# print(np.dot(y,x))
# print(np.dot(y,x).shape)
#
# x=np.array([[1,2], [3,4],[5,6]])
# y=np.array([1,2])
# print(y.shape)
# print(x.shape)
# np.dot(x,y)
# print(np.dot(x,y))
# print(np.dot(x,y).shape)

