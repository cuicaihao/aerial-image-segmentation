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


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
