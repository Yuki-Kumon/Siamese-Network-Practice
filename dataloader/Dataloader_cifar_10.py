# -*- coding: utf-8 -*-

"""
Data loader for cifar 10
Author :
    Yuki Kumon
Last Update :
    2019-08-04
"""


from torch.utils.data import Dataset
from torchvision import transforms
# import torch.nn as nn  # ネットワーク構築用
# import torch.optim as optim  # 最適化関数
# import torch.nn.functional as F  # ネットワーク用の様々な関数
import torch.utils.data  # データセット読み込み関連

import numpy as np
from PIL import Image

import os


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class MyDataset(Dataset):
    '''
    dataset class
    '''

    def __init__(self, dict, trans1=None, same_rate=0.5):
        self.data = dict[b'data']
        self.label = dict[b'labels']
        self.trans1 = trans1
        self.same_rate = same_rate

    def __len__(self):
        return len(self.label)

    def image_reshape(self, data):
        r = np.reshape(data[:1024], (32, 32))
        g = np.reshape(data[1024:2048], (32, 32))
        b = np.reshape(data[2048:], (32, 32))
        img = np.empty([32, 32, 3])
        img[:, :, 0] = r
        img[:, :, 1] = g
        img[:, :, 2] = b
        img = img.astype(int)
        img = Image.fromarray(np.uint8(img))
        return img

    def __getitem__(self, idx):
        data = self.data[idx]
        img = self.image_reshape(data)

        label = self.label[idx]

        # same_rateの割合で、同じラベルの画像を取ってくる
        if np.random.choice([True, False], p=[self.same_rate, 1 - self.same_rate]):
            same_indices = np.where(np.array(self.label) == label)[0]
            same_idx = np.random.choice(same_indices, 1)[0]
            data = self.data[same_idx]
            img2 = self.image_reshape(data)
            same_label = 1
        else:
            different_indices = np.where(np.array(self.label) != label)[0]
            different_idx = np.random.choice(different_indices, 1)[0]
            data = self.data[different_idx]
            img2 = self.image_reshape(data)
            same_label = 0

        if self.trans1:
            img = self.trans1(img)
            img2 = self.trans1(img2)

        return img, img2, same_label


def set_trans():
    trans1 = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return trans1


def Dataloader_cifar_10(cifar_path, batch_size=16, shuffle=True):
    '''
    DataLoader for cifar 10
    リストで返す
    '''

    trans1 = set_trans()
    train_loader = []

    # sat path
    train_path = []
    for i in range(1, 6):
        train_path.append(os.path.join(cifar_path, 'data_batch_' + str(i)))
    test_path = os.path.join(cifar_path, 'test_batch')

    # set dataloader
    for i in train_path:
        train_loader.append(torch.utils.data.DataLoader(MyDataset(unpickle(i), trans1), batch_size, shuffle))
    test_loader = torch.utils.data.DataLoader(MyDataset(unpickle(test_path), trans1), batch_size, shuffle)

    return train_loader, test_loader


if __name__ == '__main__':
    """
    sanity check
    """
    data_path = '/Users/yuki_kumon/Documents/python/Siamese-Network-Practice/data/cifar-10-batches-py'

    # sataset sanity check
    cifar_dict = unpickle(data_path + '/data_batch_1')

    trans1 = set_trans()
    dataset = MyDataset(cifar_dict, trans1)
    dataset[100]

    # DataLoader sanity check
    loader1, _ = Dataloader_cifar_10(data_path)
    for (img, img2, label) in (loader1[4]):
        dummy = 0
    print("sanity check is passed")
