import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np


class ConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.core_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x = self.core_layers(x)
        x = self.maxpool(x)
        return x
    
class ConvBlock3(nn.Module):
    def __init__(self, in_channels, out_channels, pool_kernel=2, pool_stride=2, ceil_mode=False):
        super().__init__()
        self.core_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride, ceil_mode=ceil_mode)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        in_medias_res = self.core_layers(x)
        x = self.maxpool(in_medias_res)
        return x, in_medias_res



class truncated_VGG16(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.layer1 = ConvBlock2(in_channels, 64)
        self.layer2 = ConvBlock2(64, 128)
        self.layer3 = ConvBlock3(128, 256, ceil_mode=True)
        self.layer4 = ConvBlock3(256, 512)
        self.layer5 = ConvBlock3(512, 512, pool_kernel=3, pool_stride=1)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        
    def forward(self, x):
        # Input in the original paper is a 224x224 RGB image
        x = self.layer1(x)
        x = self.layer2(x)
        x, _ = self.layer3(x)
        x, in_medias_res = self.layer4(x)
        x, _ = self.layer5(x)
        return x, in_medias_res
        
        
        
        
        
        