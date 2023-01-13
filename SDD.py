import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from VGG16 import VGG


class ScaleMap(nn.Module):
    def __init__(self, channels, box_num, class_num):
        super(ScaleMap, self).__init__()
        if len(channels) != 3:
            raise ValueError('ConvBlock must have exactly 3 channel values')
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=1, stride=1)
        self.norm1 = nn.BatchNorm2d(channels[1])
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1)
        self.norm2 = nn.BatchNorm2d(channels[2])
        self.feature_map = nn.Conv2d(channels[2], box_num * (4 + class_num), kernel_size=3, stride=1)

    def _init_weights(self, module):
        pass

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        features = self.feature_map(x)
        return x, features


class SSD(nn.Module):
    def __init__(self, base='vgg', class_num=2):
        super(SSD, self).__init__()
        if base == 'vgg':
            self.base_network = VGG()
        # elif base == 'inception':
        #    self.base_network = Inception()
        else:
            raise ValueError('SSD base network')
        # 6 escalas
        vgg_out = 42
        self.scale_one = nn.Conv2d(vgg_out, 4 * (4 + class_num), kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(vgg_out, 1024, kernel_size=3, stride=1)
        self.norm1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.norm2 = nn.BatchNorm2d(1024)
        self.scale_two = nn.Conv2d(1024, 6 * (4 + class_num), kernel_size=3, stride=1)
        self.scale_three = ScaleMap((1024, 256, 512), 6, class_num)
        self.scale_four = ScaleMap((512, 128, 256), 6, class_num)
        self.scale_five = ScaleMap((256, 128, 256), 4, class_num)
        self.scale_six = ScaleMap((256, 128, 256), 4, class_num)

    def _init_weights(self, module):
        pass

    def forward(self, x):
        x = self.base_network(x)
        scale_one_out = self.scale_one(x)
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        scale_two_out = self.scale_two(x)
        x, scale_three_out = self.scale_three(x)
        x, scale_four_out = self.scale_four(x)
        x, scale_five_out = self.scale_five(x)
        x, scale_six_out = self.scale_six(x)
        # Asumimos que la prediccion es una lista de tensores (c + 4) x DBs x H x W para cada escala
        #   Para cada escala: (n_cats + 4) * n_default_boxes * H_scale * W_scale
        pass
