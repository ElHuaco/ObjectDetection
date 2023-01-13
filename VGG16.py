import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np


class convblock(nn.module):
    def __init__(self, in_channels, out_channels, pooling=False):
        super().__init__()
        self.core_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.apply(self._init_weights)
        self.pooling = pooling

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
        if self.pooling:
            return self.maxpool(x)
        else:
            return x



class truncated_VGG16(nn.Module):
    def __init__(self, in_channels, n_layers=5):
        super().__init__()
        
        self.layer1 = convblock(in_channels, 64, pooling=True)
        self.layer2 = convblock(64, 128, pooling=True)
        self.layer3 = convblock(128, 256)
        self.layer4 = convblock(256, 256, pooling=True)
        self.layer5 = convblock(256, 512)
        self.layer6 = convblock(512, 512, pooling=True)
        self.layer7 = convblock(512, 512)
        self.layer8 = convblock(512, 512, pooling=True)
        
        self.n_layers = n_layers
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
        layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7, self.layer8]
        for layer in layers[:self.n_layers]:
            x = layer(x)
        return x
        
        
        
        
        
        