#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Created on   :2021/02/18 20:28:07
@author      :Caihao (Chris) Cui
@file        :loss.py
@content     :xxx xxx xxx
@version     :0.1
@License :   (C)Copyright 2020 MIT
'''

# here put the import lib

# Source: https://github.com/ycszen/pytorch-segmentation/blob/master/loss.py


import torch
import torch.nn.functional as F


class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self, weight=None, reduction="mean"):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = torch.nn.NLLLoss(weight, reduction=reduction)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)
