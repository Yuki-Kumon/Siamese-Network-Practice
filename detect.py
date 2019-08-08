# -*- coding: utf-8 -*-

"""
Detect
Author :
    Yuki Kumon
Last Update :
    2019-08-07
"""


# from torch.utils.data import Dataset
# from torchvision import transforms
# import torch.nn as nn  # ネットワーク構築用
import torch.optim as optim  # 最適化関数
# import torch.nn.functional as F  # ネットワーク用の様々な関数
import torch.utils.data  # データセット読み込み関連

from models.Siamese_Network import Siamese_Network
from models.ContrastiveLoss import ContrastiveLoss
from dataloader.Dataloader_cifar_10 import Dataloader_cifar_10

import numpy as np
# from PIL import Image

# from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt

from absl import app, flags, logging
from absl.flags import FLAGS


flags.DEFINE_string('cifar_path', './data/cifar-10-batches-py', 'cifar 10 dataset path')
flags.DEFINE_integer('input_image_size', 32, 'input imagsize(need to divided by 2**depth)')
flags.DEFINE_integer('output_len', 2, 'size after dense layer')
flags.DEFINE_float('same_rate', 0.5, 'positive rate when training')
flags.DEFINE_integer('depth', 1, 'CNN depth')
flags.DEFINE_float('dropout_ratio', 0.1, 'CNN dropout ratio')
flags.DEFINE_integer('epoch', 20, 'epoch number')
flags.DEFINE_float('lr', 0.01, 'learning rate')
flags.DEFINE_float('momentum', 0.5, 'momentum')
flags.DEFINE_bool('is_cuda', False, 'whether cuda is used or not')
flags.DEFINE_bool('pre_trained', True, 'whether model is pretrained or not')
flags.DEFINE_string('model_path', './output/model.tar', 'model path')
flags.DEFINE_string('tensorboard_path', './tensorboard_log', 'tensorboardX logging dir')
flags.DEFINE_string('eval_path', './output/output.png', 'evaluation image save path')


def main(_argv):
    # TENSOR_BOARD_LOG_DIR = FLAGS.tensorboard_path
    # writer = SummaryWriter(TENSOR_BOARD_LOG_DIR)

    # set cuda flag
    is_cuda = FLAGS.is_cuda
    if is_cuda:
        logging.info('cuda is used')
    else:
        logging.info('cuda is not used')

    # set model
    model = Siamese_Network(input_channels=3,
                            depth=FLAGS.depth,
                            input_size=FLAGS.input_image_size,
                            output_len=FLAGS.output_len,
                            dropout_ratio=FLAGS.dropout_ratio)
    if is_cuda:
        model = model.to('cuda')
    logging.info('model is created')

    # set criterion and optimizer
    optimizer = optim.SGD(model.parameters(), lr=FLAGS.lr, momentum=FLAGS.momentum)
    criterion = ContrastiveLoss()
    if(is_cuda):
        criterion = criterion.to('cuda')
    logging.info('optimizer and criterion are created')

    # if pretrained, load checkpoint
    if(FLAGS.pre_trained):
        path = FLAGS.model_path
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_old = checkpoint['epoch']
        logging.info('checkpoint is loaded')
    else:
        epoch_old = 0
        logging.info('checkpoint is NOT loaded')

    # set dataloader
    _, test_loader = Dataloader_cifar_10(FLAGS.cifar_path, shuffle=False)
    logging.info('dataloader is created')

    # define evaluation function
    def eval():
        '''
        evalate function
        2次元空間にプロットすることで分類を確かめる
        '''
        # output_list = []
        # label_list = []
        model.eval()
        for batch_idx, (image1, label) in enumerate(test_loader):
            # forwadr
            optimizer.zero_grad()
            if is_cuda:
                image1 = image1.to('cuda')
                label = label.to('cuda')
            output = model.forward_once(image1)
            if is_cuda:
                image1 = image1.to('cpu')
                label = label.to('cpu')
            if 'outputs' in locals():
                outputs = image1
            # output_list.append(output.detach().numpy())
            # label_list.append(label.detach().numpy())

        logging.info('evaluation is finished')
        return np.array(output_list), np.array(label_list)

    # execute evaluation
    outputs, labels = eval()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
