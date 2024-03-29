import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from VGG16 import VGGBase
from utils import matching, create_all_boxes, offsets2coords

# SSD model

# scale maps to obtain intermediate feature maps box coordinates and offsets
class ScaleMap(nn.Module):
    def __init__(self, channels, box_num, class_num, second_stride, kernel_size=3, second_pad=1, norm=True):
        super(ScaleMap, self).__init__()
        if len(channels) != 3:
            raise ValueError('ConvBlock must have exactly 3 channel values')
        self.box_num = box_num
        self.norm = norm
        self.class_num = class_num
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=1, stride=1)
        self.norm1 = nn.BatchNorm2d(channels[1])
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=second_stride, padding=second_pad)
        self.norm2 = nn.BatchNorm2d(channels[2])
        self.offset = nn.Conv2d(channels[2], self.box_num * 4, kernel_size=kernel_size, stride=1, padding='same')
        self.confid = nn.Conv2d(channels[2], self.box_num * self.class_num, kernel_size=kernel_size, stride=1, padding='same')

    def _init_weights(self, module):        
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        if self.norm:
            x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        if self.norm:
            x = self.norm2(x)
        x = F.relu(x)
        _, _, h, w = x.size()
        # Prediction offset shape: Batch x (H * W * Priors) x 4
        offset = torch.reshape(self.offset(x), (-1, h * w * self.box_num, 4))
        # Prediction offset shape: Batch x (H * W * Priors) x Classes
        confid = torch.reshape(self.confid(x), (-1, h * w * self.box_num, self.class_num))
        return x, offset, confid


# main model
class SSDmodel(nn.Module):
    def __init__(self, in_channels=3, base='vgg', class_num=2):
        super(SSDmodel, self).__init__()
        # Default boxes
        self.predefined_boxes = create_all_boxes()
        
        # Base architecture
        if base == 'vgg':
            self.base_network = VGGBase()
        else:
            raise ValueError('SSD base network')
        
        self.class_num = class_num
        vgg_out = 512
        self.scale1_offs = nn.Conv2d(vgg_out, 4 * 4, kernel_size=3, stride=1, padding='same')
        self.scale1_conf = nn.Conv2d(vgg_out, 4 * self.class_num, kernel_size=3, stride=1, padding='same')
        self.conv1 = nn.Conv2d(vgg_out, 1024, kernel_size=3, padding=6, dilation=6)
        self.norm1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(1024)
        self.scale2_offs = nn.Conv2d(1024, 6 * 4, kernel_size=3, stride=1, padding='same')
        self.scale2_conf = nn.Conv2d(1024, 6 * self.class_num, kernel_size=3, stride=1, padding='same')
        self.scale3 = ScaleMap((1024, 256, 512), 6, self.class_num, second_stride=2)
        self.scale4 = ScaleMap((512, 128, 256), 6, self.class_num, second_stride=2)
        self.scale5 = ScaleMap((256, 128, 256), 4, self.class_num, second_stride=1, second_pad=0)
        self.scale6 = ScaleMap((256, 128, 256), 4, self.class_num, second_stride=1, kernel_size=1, second_pad=0, norm=False)

    def _init_weights(self, module):        
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module)
            if module.bias is not None:
                module.bias.data.zero_()

    def get_default_boxes(self):
        return self.predefined_boxes

    def forward(self, x):
        in_medias_res, x = self.base_network(x)
        _, _, h, w = in_medias_res.size()
        scale1_offs = torch.reshape(self.scale1_offs(in_medias_res), (-1, h * w * 4, 4))
        scale1_conf = torch.reshape(self.scale1_conf(in_medias_res), (-1, h * w * 4, self.class_num))
        _, _, h, w = x.size()
        scale2_offs = torch.reshape(self.scale2_offs(x), (-1, h * w * 6, 4))
        scale2_conf = torch.reshape(self.scale2_conf(x), (-1, h * w * 6, self.class_num))
        x, scale3_offs, scale3_conf = self.scale3(x)
        x, scale4_offs, scale4_conf = self.scale4(x) 
        x, scale5_offs, scale5_conf = self.scale5(x)
        x, scale6_offs, scale6_conf = self.scale6(x)
        offs = torch.cat((scale1_offs, scale2_offs, scale3_offs, scale4_offs, scale5_offs, scale6_offs), dim=1)
        coords = offsets2coords(offs, self.predefined_boxes)
        conf = torch.cat((scale1_conf, scale2_conf, scale3_conf, scale4_conf, scale5_conf, scale6_conf), dim=1)
        conf = F.softmax(conf, dim=2)
        return coords, conf

    def predict(self, x, min_conf=0.01, max_overlap=0.45, top=200):
        coords, conf = self.forward(x)
        pred_coords = list([torch.empty(size=(1,4)).cuda()] * coords.shape[0])
        pred_conf = list([torch.empty(size=(1,1)).cuda()] * conf.shape[0])
        pred_labels = list([torch.empty(size=(1,1), dtype=torch.int32).cuda()] * conf.shape[0])
        for b in range(coords.shape[0]):
            for c in range(self.class_num):
                class_conf = conf[b, :, c]
                class_conf_sorted, indeces = torch.sort(class_conf, dim=0, descending=True)
                is_prediction = torch.ones(class_conf.shape, dtype=torch.bool).cuda() # Order here is the sorted indeces
                for row, pred in enumerate(class_conf_sorted[:-1]):
                    if pred > min_conf:          
                        non_overlapping = torch.logical_not(matching(coords[b, indeces[row], :].unsqueeze(0),
                                                                     coords[b, indeces[row+1:], :],
                                                                     threshold=max_overlap) )
                        is_prediction[row+1:] = torch.logical_and(is_prediction[row+1:],
                                                                              non_overlapping.squeeze(0))
                    else:
                        is_prediction[row] = False
                indeces = indeces[is_prediction][:top]
                pred_conf[b] = torch.cat((pred_conf[b], class_conf[indeces].unsqueeze(0).transpose(1, 0)))
                pred_coords[b] = torch.cat((pred_coords[b], coords[b, indeces, :]))
                pred_labels[b] = torch.cat((pred_labels[b],
                                            c * torch.ones(size=(len(indeces),1), dtype=torch.int32).cuda()))
            pred_coords[b] = pred_coords[b][1:, :]
            pred_conf[b] = pred_conf[b][1:, :]
            pred_labels[b] = pred_labels[b][1:, :]
        return pred_coords, pred_conf, pred_labels
