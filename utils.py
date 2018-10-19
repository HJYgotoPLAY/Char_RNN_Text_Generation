# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf

def get_batches(arr, n_seqs, n_steps):
    """
    对已有数组进行mini-batch分割
    :param arr: 待分割的数组
    :param n_seqs: 一个batch中的序列个数
    :param n_steps: 单个序列长度
    """
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    # 仅保留完整的batch，舍弃不能整除的部分
    arr = arr[: batch_size * n_batches]
    # 重塑
    arr = arr.reshape((n_seqs, -1))
    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n : n+n_steps]
        # targets相比于x向后错位一个字符
        y = np.zeros_like(x)
        y[:,:-1] = x[:,1:]
        y[:,-1] = y[:,0]
        yield x, y
