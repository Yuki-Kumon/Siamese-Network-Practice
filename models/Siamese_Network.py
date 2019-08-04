# -*- coding: utf-8 -*-

"""
Siamese Network on pytorch
Author :
    Yuki Kumon
Last Update :
    2019-08-04
"""


# from torch.utils.data import Dataset
# from torchvision import transforms
import torch.nn as nn  # ネットワーク構築用
# import torch.optim as optim  # 最適化関数
import torch.nn.functional as F  # ネットワーク用の様々な関数
import torch.utils.data  # データセット読み込み関連

import numpy as np
# from PIL import Image

# from torchsummary import summary
# from tensorboardX import SummaryWriter

# import matplotlib.pyplot as plt


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    '''
    convolution layer
    '''
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


class Conv_layer(nn.Module):
    '''
    畳み込み層
    '''

    def __init__(self, in_channels, out_channels, dropout_ratio):
        # initialization of class
        super(Conv_layer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = conv3x3(in_channels, out_channels)
        self.dropout = nn.Dropout2d(p=dropout_ratio)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        # convolution
        x = F.relu(self.batchnorm(self.conv(x)))
        # dropout
        x = self.dropout(x)

        return x


class CNN(nn.Module):
    '''
    Siamese Networkで用いる畳み込みニューラルネットワーク
    '''

    def __init__(self, input_channels=3, depth=5, dropout_ratio=0.1):
        # initialization of class
        super(CNN, self).__init__()

        # create layers
        self.layers = []
        for i in range(depth):
            output_channnels = 2 ** (i + 4)
            layer = Conv_layer(input_channels, output_channnels, dropout_ratio)
            self.layers.append(layer)
            input_channels = output_channnels

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for module in self.layers:
            x = module(x)

        return x


class Siamese_Network(nn.Module):
    '''
    Siamese Network
    '''

    def __init__(self, input_channels=3, depth=5, dropout_ratio=0.1):
        # initialization of class
        super(Siamese_Network, self).__init__()

        # create CNN
        self.cnn = CNN(input_channels, depth, dropout_ratio)

    def forward(self, x0, x1):
        return self.cnn(x0), self.cnn(x1)


if __name__ == '__main__':
    """
    Sanity Check
    """

    """
    img = torch.from_numpy(np.random.rand(4, 3, 256, 256)).float()

    # sanity check for Conv_layer
    conv = Conv_layer(3, 16, 0.1)
    nn.init.normal_(conv.conv.weight, 0.0, 1.0)
    conv.eval()

    # print(img.size())
    out = conv(img)
    print('sanity check for \"Conv_layer\" is passed')

    # sanity check for CNN
    cnn = CNN()
    out = cnn(img)
    print('sanity check for \"CNN\" is passed')
    """

    from torchsummary import summary

    conv = Conv_layer(3, 16, 0.1)
    summary(conv, (3, 256, 256))

    cnn = CNN()
    summary(cnn, (3, 256, 256))

    sim = Siamese_Network()
    img1 = torch.from_numpy(np.random.rand(4, 3, 256, 256)).float()
    img2 = torch.from_numpy(np.random.rand(4, 3, 256, 256)).float()
    _, _ = sim(img1, img2)

    print("sanity chack is passed!")
