import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from VGG16 import truncated_VGG16


class ScaleMap(nn.Module):
    def __init__(self, channels, box_num, class_num, second_stride):
        super(ScaleMap, self).__init__()
        if len(channels) != 3:
            raise ValueError('ConvBlock must have exactly 3 channel values')
        self.box_num = box_num
        self.class_num = class_num
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=1, stride=1)
        self.norm1 = nn.BatchNorm2d(channels[1])
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=second_stride)
        self.norm2 = nn.BatchNorm2d(channels[2])
        self.offset = nn.Conv2d(channels[2], self.box_num * 4, kernel_size=3, stride=1)
        self.confid = nn.Conv2d(channels[2], self.box_num * self.class_num, kernel_size=3, stride=1)

    def _init_weights(self, module):
        pass

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        _, _, h, w = x.size
        # Prediction offset shape: Batch x (H * W * Priors) x 4
        offset = torch.reshape(self.offset(x), (-1, h * w * self.box_num, 4))
        # Prediction offset shape: Batch x (H * W * Priors) x Classes
        confid = torch.reshape(self.confid(x), (-1, h * w * self.box_num, self.class_num))
        return x, offset, confid


class SSD(nn.Module):
    def __init__(self, in_channels=3, base='vgg', class_num=2):
        super(SSD, self).__init__()
        if base == 'vgg':
            self.base_network = truncated_VGG16(in_channels)
        # elif base == 'inception':
        #    self.base_network = Inception()
        else:
            raise ValueError('SSD base network')
        self.class_num = class_num
        vgg_out = 42
        self.scale1_offs = nn.Conv2d(vgg_out, 4 * 4, kernel_size=3, stride=1)
        self.scale1_conf = nn.Conv2d(vgg_out, 4 * self.class_num, kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(vgg_out, 1024, kernel_size=3, stride=1)
        self.norm1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.norm2 = nn.BatchNorm2d(1024)
        self.scale2_offs = nn.Conv2d(1024, 6 * 4, kernel_size=3, stride=1)
        self.scale2_conf = nn.Conv2d(1024, 6 * self.class_num, kernel_size=3, stride=1)
        self.scale3 = ScaleMap((1024, 256, 512), 6, self.class_num, second_stride=2)
        self.scale4 = ScaleMap((512, 128, 256), 6, self.class_num, second_stride=2)
        self.scale5 = ScaleMap((256, 128, 256), 4, self.class_num, second_stride=1)
        self.scale6 = ScaleMap((256, 128, 256), 4, self.class_num, second_stride=1)

    def _init_weights(self, module):
        pass

    def forward(self, x):
        x = self.base_network(x)
        _, _, h, w = x.size
        scale1_offs = torch.reshape(self.scale1_offs(x), (-1, h * w * 4, 4))
        scale1_conf = torch.reshape(self.scale1_conf(x), (-1, h * w * 4, self.class_num))
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        _, _, h, w = x.size
        scale2_offs = torch.reshape(self.scale2_offs(x), (-1, h * w * 6, 4))
        scale2_conf = torch.reshape(self.scale2_conf(x), (-1, h * w * 6, self.class_num))
        x, scale3_offs, scale3_conf = self.scale3(x)
        x, scale4_offs, scale4_conf = self.scale4(x)
        x, scale5_offs, scale5_conf = self.scale5(x)
        _, scale6_offs, scale6_conf = self.scale6(x)
        offs = torch.cat((scale1_offs, scale2_offs, scale3_offs, scale4_offs, scale5_offs, scale6_offs), dim=1)
        conf = torch.cat((scale1_conf, scale2_conf, scale3_conf, scale4_conf, scale5_conf, scale6_conf), dim=1)
        return offs, conf
