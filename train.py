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
from models.ContrastiveLoss import ContrastiveLoss
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
flags.DEFINE_float('lr', 0.01, 'learning rate')
flags.DEFINE_float('momentum', 0.5, 'momentum')
flags.DEFINE_bool('is_cuda', False, 'whether cuda is used or not')
flags.DEFINE_bool('pre_trained', False, 'whether model is pretrained or not')
flags.DEFINE_string('model_path', './model.tar', 'model path')
flags.DEFINE_string('tensorboard_path', './tensorboard_log', 'tensorboardX logging dir')


def main(_argv):
    TENSOR_BOARD_LOG_DIR = FLAGS.tensorboard_path
    writer = SummaryWriter(TENSOR_BOARD_LOG_DIR)

    # set cuda flag
    is_cuda = FLAGS.is_cuda
    if is_cuda:
        logging.info('cuda is used')
    else:
        logging.info('cuda is not used')

    # set model
    model = Siamese_Network(input_channels=3, depth=FLAGS.depth, dropout_ratio=FLAGS.dropout_ratio)
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
    train_loader, test_loader = Dataloader_cifar_10(FLAGS.cifar_path)
    logging.info('dataloader is created')

    # define training function
    def train(epoch, train_loader_idx):
        '''
        train function
        '''
        model.train()
        for batch_idx, (image1, image2, label) in enumerate(train_loader[train_loader_idx]):
            # forwadr
            optimizer.zero_grad()
            if is_cuda:
                image1 = image1.to('cuda')
                image2 = image2.to('cuda')
                label = label.to('cuda')
            output1, output2 = model(image1, image2)
            loss = criterion(output1, output2, label)
            # print(output)
            # backward
            loss.backward()
            optimizer.step()
            # print(output.data.max(1)[1])
            # print(label)
            """
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image), len(loader.dataset),
                100. * batch_idx / len(loader), loss.item()))
            """
        writer.add_scalar('train_loss', loss.item(), epoch)
        # print('train epoch ', str(epoch), ', loss: ', str(loss.item()))
        logging.info('train epoch {}, loss: {}'.format(epoch, loss.item()))

    # execute training
    logging.info('start training')
    for epoch_num in range(epoch_old, epoch_old + FLAGS.epoch):
        loader_num = np.random.choice([0, 1, 2, 3, 4])
        train(epoch_num, loader_num)

    # save
    torch.save({
        'epoch': epoch_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    },
        FLAGS.model_path
    )
    logging.info('trained model is saved')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
