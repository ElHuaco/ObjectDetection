import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np


# Las predicciones traducidas a relative-space desde su escala, se matchean con Jaccard a cada ground truth
# Se devuelven aquellas que superen 0.5

def matching(predicted_boxes, target_boxes, threshold=0.5):

    # Jaccard coefficient measures similarity between finite sample sets, 
    # it is defined as the size of the intersection divided by the size of the union of the sample sets

    '''
    input: predicted_boxes - (N, 4): (n_boxes, (x_min, y_min, x_max, y_max))
    input: target_boxes - (M, 4): (n_boxes, (x_min, y_min, x_max, y_max))

    output: matching - (N, M) - 0 for no match, 1 for match
    '''

    # Calcular la intersección
    # Create 4 tensors (N, M, 1) that compare the 4 values for each pair of boxes
    inter_x1 = torch.max(predicted_box[:, None, 0], target_box[None, :, 0])
    inter_y1 = torch.max(predicted_box[:, None, 1], target_box[None, :, 1])
    inter_x2 = torch.min(predicted_box[:, None, 2], target_box[None, :, 2])
    inter_y2 = torch.min(predicted_box[:, None, 3], target_box[None, :, 3])

    # Obtain the dimensions of the intersection by getting 'x_max - x_min', 'y_max - y_min'. If it's negative, set to 0.
    inter_W = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_H = torch.clamp(inter_y2 - inter_y1, min=0)

    # Obtain the area
    A_inter = inter_H*inter_W

    # Area of boxes
    A_predicted = (predicted_box[:, 2] - predicted_box[:, 0])*(predicted_box[:, 3] - predicted_box[:, 1])
    A_target = (target_box[:, 2] - target_box[:, 0])*(target_box[:, 3] - target_box[:, 1])

    # Union area
    A_union = A_predicted + A_target - A_inter

    # Jaccard similarity
    jaccard = A_inter/A_union
    
    # Find the best matching ground truth box for each predicted box
    best_gt = torch.argmax(jaccard, dim=1)

    # Matching value en función de la Jaccard similarity
    matches = (jaccard > threshold).float()
    
    return matches, best_gt

    
def hard_mining(matches, confidences, ratio=3):
    
    # Mask the matches to keep them
    positives = confidences[matches]
    
    # Find the negative matches
    mask_negatives = torch.logical_not(matches)
    negatives = confidences[mask_negatives]
    
    # Sort the negative examples by confidence score in descending order
    sorted_negatives, indices = torch.sort(negatives, descending=True)
    
    # Take the top ratio*(num_positives)
    n_positives = positives.size(dim=1).int()
    keep = sorted_negatives[:(ratio*n_positives)]
    
    # return n_positives bc we will need it for the loss
    #torch.cat(positives, keep)
    return positives, keep, n_positives
    