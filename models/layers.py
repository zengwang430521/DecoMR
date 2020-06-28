"""
This file contains definitions of layers used as building blocks in DPNet and UVNet
"""
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCBlock(nn.Module):
    """Wrapper around nn.Linear that includes batch normalization and activation functions."""
    def __init__(self, in_size, out_size, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(FCBlock, self).__init__()
        module_list = [nn.Linear(in_size, out_size)]
        if batchnorm:
            module_list.append(nn.BatchNorm1d(out_size))
        if activation is not None:
            module_list.append(activation)
        if dropout:
            module_list.append(dropout)
        self.fc_block = nn.Sequential(*module_list)
        
    def forward(self, x):
        return self.fc_block(x)


class FCResBlock(nn.Module):
    """Residual block using fully-connected layers."""
    def __init__(self, in_size, out_size, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(FCResBlock, self).__init__()
        self.fc_block = nn.Sequential(nn.Linear(in_size, out_size),
                                      nn.BatchNorm1d(out_size),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(out_size, out_size),
                                      nn.BatchNorm1d(out_size))
        
    def forward(self, x):
        return F.relu(x + self.fc_block(x))


class ConvBottleNeck(nn.Module):
    """
    the Bottleneck Residual Block in ResNet
    """

    def __init__(self, in_channels, out_channels, nl_layer=nn.ReLU(inplace=True), norm_type='GN'):
        super(ConvBottleNeck, self).__init__()
        self.nl_layer = nl_layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)

        if norm_type == 'BN':
            affine = True
            # affine = False
            self.norm1 = nn.BatchNorm2d(out_channels // 2, affine=affine)
            self.norm2 = nn.BatchNorm2d(out_channels // 2, affine=affine)
            self.norm3 = nn.BatchNorm2d(out_channels, affine=affine)
        elif norm_type == 'SYBN':
            affine = True
            # affine = False
            self.norm1 = nn.SyncBatchNorm(out_channels // 2, affine=affine)
            self.norm2 = nn.SyncBatchNorm(out_channels // 2, affine=affine)
            self.norm3 = nn.SyncBatchNorm(out_channels, affine=affine)
        else:
            self.norm1 = nn.GroupNorm((out_channels // 2) // 8, (out_channels // 2))
            self.norm2 = nn.GroupNorm((out_channels // 2) // 8, (out_channels // 2))
            self.norm3 = nn.GroupNorm(out_channels // 8, out_channels)

        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):

        residual = x

        y = self.conv1(x)
        y = self.norm1(y)
        y = self.nl_layer(y)

        y = self.conv2(y)
        y = self.norm2(y)
        y = self.nl_layer(y)

        y = self.conv3(y)
        y = self.norm3(y)

        if self.in_channels != self.out_channels:
            residual = self.skip_conv(residual)
        y += residual
        y = self.nl_layer(y)
        return y


# A net similar to hourglass, used in the UV net (Location Net)
class HgNet(nn.Module):
    def __init__(self, in_channels, level, nl_layer=nn.ReLU(inplace=True), norm_type='GN'):
        super(HgNet, self).__init__()

        down_layers = []
        up_layers = []
        if norm_type == 'GN':
            self.norm = nn.GroupNorm(in_channels // 8, in_channels)
        elif norm_type == 'BN':
            affine = True
            self.norm = nn.BatchNorm2d(in_channels, affine=affine)

        for i in range(level):
            out_channels = in_channels * 2

            down_layers.append(
                nn.Sequential(
                    ConvBottleNeck(in_channels=in_channels, out_channels=out_channels, nl_layer=nl_layer, norm_type=norm_type),
                    nn.MaxPool2d(kernel_size=2, stride=2)))
            up_layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    ConvBottleNeck(in_channels=out_channels, out_channels=in_channels, nl_layer=nl_layer, norm_type=norm_type)))

            in_channels = out_channels

        self.down_layers = nn.ModuleList(down_layers)
        self.up_layers = nn.ModuleList(up_layers)

    def forward(self, x):

        feature_list = []
        y = x
        for i in range(len(self.down_layers)):
            feature_list.append(y)
            y = self.down_layers[i](y)

        for i in range(len(self.down_layers) - 1, -1, -1):
            y = self.up_layers[i](y) + feature_list[i]

        y = self.norm(y)
        return y


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def deconv3x3(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Upsample(scale_factor=stride, mode='bilinear'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_planes, out_planes,
                  kernel_size=3, stride=1, padding=0)
    )
    # return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=stride, padding=1, bias=False)

