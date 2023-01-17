import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

# Useful functions for training SSD


def matching(predicted_boxes, target_boxes, threshold=0.5):

    # Relaciona las ground truth boxes con las predicted boxes;
    # llamamos negativos a todos los pares cuya Jaccard similarity < 0.5
    
    '''
    input: predicted_boxes - (N, 4) - top-left and bottom-right coordinates of the box - (n_boxes, (x1, y1, x2, y2))
    input: target_boxes - (M, 4) - top-left and bottom-right coordinates of the box - (n_boxes, (x1, y1, x2, y2))

    output: matching - (N, M) - boolean type 
    '''

    # Calcular la intersección
    # Create 4 tensors (N, M, 1) that compare the 4 values for each pair of boxes
    # This gives us the top-left and bottom-right corners of the intersection
    inter_x1 = torch.max(predicted_boxes[:, None, 0], target_boxes[None, :, 0])
    inter_y1 = torch.max(predicted_boxes[:, None, 1], target_boxes[None, :, 1])
    inter_x2 = torch.min(predicted_boxes[:, None, 2], target_boxes[None, :, 2])
    inter_y2 = torch.min(predicted_boxes[:, None, 3], target_boxes[None, :, 3])

    # Obtain the dimensions of the intersection. If it's negative, no intersection -> set to 0.
    inter_W = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_H = torch.clamp(inter_y2 - inter_y1, min=0)

    # Obtain the area for each pair of boxes (element-wise multiplication)
    A_inter = inter_H*inter_W # size (N, M, 1)

    # Area of boxes
    A_predicted = (predicted_box[:, 2] - predicted_box[:, 0])*(predicted_box[:, 3] - predicted_box[:, 1]) # size (N, 1)
    A_target = (target_box[:, 2] - target_box[:, 0])*(target_box[:, 3] - target_box[:, 1]) # size (M, 1)

    # Union area - broadcast values in A_predicted to add each of the areas to every area in A_target
    A_union = A_predicted[:, None] + A_target - A_inter # size (N, M, 1)

    # Jaccard coef: intersection divided by the union of the sample sets - measures similarity between 2 finite sets
    # elementwise division to get the jaccard similarity for every pair of boxes
    jaccard = A_inter/A_union 

    # Matching value en función de la Jaccard similarity y el threshold
    matches = (jaccard > threshold)
    
    return matches


# Use predefined boxes and offsets to get the coordinates of predicted boxes in the input image scale
def offsets2coords(offsets, default_boxes):
    '''
    input: offsets - (B, 4*n_boxes, W, H) - coords in the form (cx, cy, w, h)
    input: default_boxes - (n_boxes*W*H, 4) - coords in the form (cx, cy, w, h)
    
    output: coordinates - (n_boxes, 4) - coordinates of the predicted boxes (top-left, bottom-right)
    '''
    
    predicted_boxes = torch.empty_like(offsets)
    
    # Get centers (cx, cy) of predicted boxes
    predicted_boxes[:, 2:] = offsets[:, 2:]*default_boxes[:, :2] + default_boxes[:, 2:]
    # Get w, h of predicted boxes
    predicted_boxes[:, :2] = torch.exp(offsets[:, :2])*default_boxes[:, :2]
    
    # Transform to appropriate coordinates
    coordinates = torch.empty_like(offsets)
    
    coordinates[:, 0] = predicted_boxes[:, 0] - predicted_boxes[:, 2]/2
    coordinates[:, 1] = predicted_boxes[:, 1] + predicted_boxes[:, 3]/2
    coordinates[:, 2] = predicted_boxes[:, 0] + predicted_boxes[:, 2]/2
    coordinates[:, 3] = predicted_boxes[:, 1] - predicted_boxes[:, 3]/2
    
    return coordinates


# Default boxes for training for each scale
def create_FM_boxes(aspect_ratios, scale, FM_size, extra_box_scale=None):
    '''
    input: aspect_ratios - set of aspect ratios for the default boxes
    input: scale - scale of the default boxes for the specific feature map
    input: FM_size - size of the specific feature map
    
    output: default_boxes - tensor (n_boxes*FM_size**2, 4) with the default boxes for the specific feature map and scale
    '''
    n_boxes = len(aspect_ratios)
    
    # Widths and heights of the default boxes
    widths = [scale*a**(1/2) for a in aspect_ratios]
    heights = [scale/a**(1/2) for a in aspect_ratios]
    if extra_box_scale is not None:
        widths.append(extra_box_scale)
        heights.append(extra_box_scale)
        n_boxes += 1
    
    # Build tensor for storing all the boxes coordinates
    total_boxes = FM_size**2*n_boxes
    default_boxes = torch.empty(size=(total_boxes, 4))
    
    prev_indx = 0
    i = 0
    j = 0
    # Set centers and store the coordinates
    for pix_indx in torch.arange(n_boxes, total_boxes+1, n_boxes):
        default_boxes[prev_indx:pix_indx, 2] = torch.Tensor(widths)
        default_boxes[prev_indx:pix_indx, 3] = torch.Tensor(heights)
        default_boxes[prev_indx:pix_indx, 0] = (i + 0.5)/FM_size
        default_boxes[prev_indx:pix_indx, 1] = (j + 0.5)/FM_size
        prev_indx = pix_indx
        j += 1
        if j == FM_size:
            j = 0
            i += 1

    return default_boxes


# Create all default boxes for the model
def set_scales(scale_min, scale_max, n_scales):
    scales = []
    for k in range(n_scales):
        scales.append(scale_min + (scale_max - scale_min)*k/(n_scales - 1))
    return scales

    
def create_all_boxes(FM_sizes = (8, 19, 10, 5, 3, 1)):
       
    aspect_ratios = ([1., 2., 0.5],
                     [1., 2., 3., 0.5, .333],
                     [1., 2., 3., 0.5, .333],
                     [1., 2., 3., 0.5, .333],
                     [1., 2., 0.5],
                     [1., 2., 0.5])
    
    scales = set_scales(0.2, 0.9, 6)
    default_boxes = torch.empty(size=(0, 4))
    for k in range(6):
        try:
            extra_box = (scales[k]*scales[k+1])**(1/2)
        except IndexError:
            additional_scale = 1.
            print('entered exception')
        default_boxes = torch.cat((default_boxes, create_FM_boxes(aspect_ratios = aspect_ratios[k],
                                                                  scale = scales[k], 
                                                                  FM_size = FM_sizes[k], 
                                                                  extra_box_scale=extra_box)))
    return default_boxes