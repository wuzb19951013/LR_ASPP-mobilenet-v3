"""
    Trimap generation : T-Net

Author: Zhengwei Li
Date  : 2018/12/24
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + se + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        if self.se != None:
            out = self.se(out)
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class InvertedResidual(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        # assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3,
                      stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class mobilenet_v3(nn.Module):
    def __init__(self, nInputChannels=3):
        super(mobilenet_v3, self).__init__()

        self.head_conv = nn.Sequential(nn.Conv2d(nInputChannels, 16, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(16),
                                       hswish())

        self.block_1 = Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1)
        # 1/2
        self.block_2 = nn.Sequential(
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1)
        )
        # 1/4
        self.block_3 = nn.Sequential(
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(72), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(120), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(120), 1)
        )
        # 1/8
        self.block_4 = nn.Sequential(
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1)
        )
        # 1/8
        self.block_5 = nn.Sequential(
            Block(3, 80, 480, 112, hswish(), SeModule(480), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(672), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(672), 1)
        )
        # 1/16
        self.block_6 = nn.Sequential(
            Block(5, 160, 672, 160, hswish(), SeModule(672), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(960), 1)
        )

    def forward(self, x):        
        # 1/1
        s1 = self.block_1(self.head_conv(x))
        # 1/2
        s2 = self.block_2(s1)
        # 1/4
        s3 = self.block_3(s2)
        # 1/8
        s4 = self.block_5(self.block_4(s3))
        # 1/16
        s5 = self.block_6(s4)

        return s4, s5


class T_mv2_unet(nn.Module):
    '''
        Lite R-ASPP

    '''

    def __init__(self, classes=1):

        super(T_mv2_unet, self).__init__()
        # -----------------------------------------------------------------
        # mobilenetv3
        # ---------------------
        self.feature = mobilenet_v3()

        # -----------------------------------------------------------------
        # segmentation head
        # ---------------------
        self.s1_part1 = nn.Sequential(
            InvertedResidual(160, 128, 4, 1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        )

        self.s1_part2 = nn.Sequential(
            nn.Conv2d(160, 128, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.s2_up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 16, kernel_size=1,
                      stride=1, padding=0, bias=False),
        )

        self.s2_conv = nn.Conv2d(
            160, 16, kernel_size=1, stride=1, padding=0, bias=False)

        self.s3_fusion = nn.Sequential(
            InvertedResidual(32, 16, 1, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        )

        self.last_conv_up = nn.Sequential(
            nn.Conv2d(16, classes, 3, 1, 1),
            nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, input):

        # -----------------------------------------------
        # mobilenetv3
        # ---------------------
        s2, s1 = self.feature(input)
        # -----------------------------------------------
        # segmentation head
        # ---------------------
        # SE part
        s1_1 = self.s1_part1(s1)
        s1_2 = self.s1_part2(s1)
        s1_ = s1_1*s1_2
        s1_ = self.s2_up_conv(s1_)

        # skip connection
        s2_ = self.s2_conv(s2)
        s2_ = torch.cat((s1_, s2_), 1)
        s3_ = self.s3_fusion(s2_)

        # up conv
        out = self.last_conv_up(s3_)

        return out
