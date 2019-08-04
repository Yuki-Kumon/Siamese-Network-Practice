# -*- coding: utf-8 -*-

"""
Loss function
copied from
https://vaaaaaanquish.hatenablog.com/entry/2019/02/23/214036
https://github.com/delijati/pytorch-siamese/blob/master/contrastive.py
Author :
    Yuki Kumon
Last Update :
    2019-08-04
"""


import torch


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=5.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        y = y.float()
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss
