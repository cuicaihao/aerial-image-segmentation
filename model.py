#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Created on   :2021/02/18 20:28:15
@author      :Caihao (Chris) Cui
@file        :model.py
@content     :xxx xxx xxx
@version     :0.1
@License :   (C)Copyright 2020 MIT
'''

# here put the import lib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# TODO: Use Transposed Convolution for upsampling
# https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0


class FCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(FCNN, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel :  N C H W
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=nn%20conv2d#torch.nn.Conv2d
        nc = num_classes

        # 1 Layer Features
        # 1x1 filter net-in-net
        self.conv1a = nn.Conv2d(3, 16, (1, 1))
        self.bn1a = nn.BatchNorm2d(16)
        # 3x3 filter
        self.conv1b = nn.Conv2d(3, 16, (3, 3), padding=1)
        self.bn1b = nn.BatchNorm2d(16)
        # 5x5 filter with atrous algorithm, looking large area and keep in and out same size.
        self.conv1c = nn.Conv2d(3, 16, (5, 5), stride=1, padding=4, dilation=2)
        self.bn1c = nn.BatchNorm2d(16)

        # 2 Layer Features
        self.conv2 = nn.Conv2d(48, 96, (1, 1))
        self.bn2 = nn.BatchNorm2d(96)

        # 3 Layer Features
        self.conv3 = nn.Conv2d(96, 48, (3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(48)

        # 4 Layer Features
        self.conv4 = nn.Conv2d(16 + 48, 16, (1, 1))
        self.bn4 = nn.BatchNorm2d(16)

        # 5 Layer Features
        self.conv5 = nn.Conv2d(16, nc, (3, 3), padding=1)
        self.bn5 = nn.BatchNorm2d(nc)

    def forward(self, x):
        input_size = x.size()[2:]
        # Layer 1
        xa = F.relu(self.bn1a(self.conv1a(x)))
        xb = F.relu(self.bn1b(self.conv1b(x)))
        xc = F.relu(self.bn1c(self.conv1c(x)))

        # Layer 2
        xabc = torch.cat((xa, xb, xc), 1)  # concatenated on channel
        xabc = F.relu(self.bn2(self.conv2(xabc)))
        xabc = F.max_pool2d(xabc, (2, 2))  # 0.5x

        # Layer 3
        xabc = F.relu(self.bn3(self.conv3(xabc)))
        # x = F.interpolate(x, scale_factor=(2, 2))  # 2.0x nearest neightbour lead to jaggedness
        xabc = F.interpolate(
            xabc, scale_factor=(2, 2), mode="bicubic", align_corners=True
        )  # make the image smooth and reduce sharpness

        # Layer 4
        x = torch.cat((xa, xabc), 1)
        x = self.bn4(self.conv4(x))

        # Layer 5
        x = self.bn5(self.conv5(x))
        return x
