# -*- coding: utf-8 -*-
"""
@ project: WDGRL
@ author: lzx
@ file: uitls.py
@ time: 2019/6/17 20:55
"""
import pickle
def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        print('[INFO] Object saved to {}'.format(path))

import functools
# def add(a, b):
#     return a + b

# add(4,
# plus3 = functools.partial(add, [3,5])
# plus5 = functools.partial(add, 5)
# print(plus5(4))
# sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
# gaussian_kernel = functools.partial(utils.gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
# loss_value = utils.maximum_mean_discrepancy(h_s, h_t, kernel=gaussian_kernel)
# mmd_loss = mmd_param * tf.maximum(1e-4, loss_value)
#
# def gaussian_kernel_matrix(x, y, sigmas):
#     beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
#     dist = compute_pairwise_distances(x, y)
#     s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
#     return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))
# import torch
# x = torch.randn(3,3)
# print(x,torch.diag(x))
# z_s = torch.diag(x).unsqueeze(1)
# z_s = z_s.expand_as(x)
# print(z_s)
