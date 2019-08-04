# -*- coding: utf-8 -*-

"""
Data loader for cifar 10
Author :
    Yuki Kumon
Last Update :
    2019-08-04
"""


from torch.utils.data import Dataset
# from torchvision import transforms
# import torch.nn as nn  # ネットワーク構築用
# import torch.optim as optim  # 最適化関数
# import torch.nn.functional as F  # ネットワーク用の様々な関数
# import torch.utils.data  # データセット読み込み関連

import numpy as np
from PIL import Image


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class MyDataset(Dataset):
    '''
    dataset class
    '''

    def __init__(self, dict, trans1=None):
        self.data = dict[b'data']
        self.label = dict[b'labels']
        self.trans1 = trans1

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = self.data[idx]
        r = np.reshape(data[:1024], (32, 32))
        g = np.reshape(data[1024:2048], (32, 32))
        b = np.reshape(data[2048:], (32, 32))
        img = np.empty([32, 32, 3])
        img[:, :, 0] = r
        img[:, :, 1] = g
        img[:, :, 2] = b
        img = img.astype(int)
        img = Image.fromarray(np.uint8(img))

        label = self.label[idx]

        if self.trans1:
            image = self.trans1(img)

        return image, label
