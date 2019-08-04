# -*- coding: utf-8 -*-

"""
Train
Author :
    Yuki Kumon
Last Update :
    2019-08-04
"""


from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn  # ネットワーク構築用
import torch.optim as optim  # 最適化関数
import torch.nn.functional as F  # ネットワーク用の様々な関数
import torch.utils.data  # データセット読み込み関連

from models.Siamese_Network import Siamese_Network
from dataloader.Dataloader_cifar_10 import Dataloader_cifar_10

import numpy as np
from PIL import Image

from tensorboardX import SummaryWriter

# import matplotlib.pyplot as plt

from absl import app, flags, logging
from absl.flags import FLAGS


flags.DEFINE_string('cifar_path', './data/cifar-10-batches-py', 'cifar 10 dataset path')
flags.DEFINE_float('same_rate', 0.5, 'positive rate when training')
flags.DEFINE_integer('depth', 5, 'CNN depth')
flags.DEFINE_float('dropout_ratio', 0.1, 'CNN dropout ratio')
flags.DEFINE_integer('epoch', 20, 'epoch number')
flags.DEFINE_bool('pre_trained', False, 'whether model is pretrained or not')
flags.DEFINE_string('model_path', './model.tar', 'model path')


def main(_argv):
    # set model
    Net = Siamese_Network(input_channels=3, depth=FLAGS.depth, dropout_ratio=FLAGS.dropout_ratio)

    # set dataloader
    train_loader, test_loader = Dataloader_cifar_10(FLAGS.cifar_path)

    # define training function


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
