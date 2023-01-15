# Asumimos que la predicción nos devuelve una lista de tensores (c + 4) x DBs x H x W para cada escala
# 1. Para calcular la loss, separamos los offsets y las confidences, quedándonos con dos listas
# 2. Obtenemos las coordenadas top-left y bottom-right de las pred_boxes utilizando los offsets y la escala
# 3. Relacionamos las ground truth boxes con las predicted boxes; llamamos negativos a todos los pares cuya Jaccard similarity < 0.5
# 4. Calculamos la loss.

# La loss consta de dos partes:
# - Localization loss: compara las coordenadas de los pares de cajas con relación positiva utilizando una Smooth L1
# - Confidence loss: utiliza cross entropy (log(softmax)) para:
    # - los pares positivos: utilizando la confidence correspondiente a la categoría del match
    # - las cajas de todo pares negativos: utilizando la confidence más alta, con el fin de reducirla. No se utilizarán todas, 
    # si no que se escogeran las de mayor pérdida hasta tener un radio de hasta 3:1 con los pares positivos

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def matching(predicted_boxes, target_boxes, threshold=0.5):
    
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



def SSDloss(pred_boxes, pred_confidences, gt_boxes, gt_labels, hard_mining_ratio=3, smoothL1_beta=1.0):
    '''
    input: pred_boxes - (N, 4) - 4 coordinates for every predicted box (top-left and bottom-right corners)
    input: pred_confidences - (N, num_categories) - confidence for every category in each box
    input: gt_boxes - (M, 4) - 4 coordinates for every ground truth box (top-left and bottom-right corners)
    input: gt_labels - (M, 1) - 1 value for every ground truth box, corresponding to its category (from 0 to num_categories)
    
    output: loss - total loss (confidence + location)
    '''
    
    # Define location loss type
    smoothL1 = nn.SmoothL1Loss(reduction='mean', beta=smoothL1_beta)
    
    # Compute matching between predictions and ground truth boxes
    matches = matching(pred_boxes, gt_boxes) # (N, M, 1) tensor of booleans relating predictions and GT boxes
    total_matches = matches.sum()
    
    # Get loss for every box
    matches_loss = 0
    loc_loss = 0
    negative_losses = []
    for i, pred_box in enumerate(pred_boxes):
        # For each predicted box get the matching GTs and their labels
        row_mask = matches[i,:] # should be a 1-dim tensor of length M - tipo [False, True, True, ...]
        box_matches = row_mask.sum()
        
        # Softmax of current box confidences
        conf_softmax = F.softmax(pred_confidences[i])
        
        # Check if current box matches with a ground truth
        if box_matches != 0:   
            # if it has one or more matches, find to which GT boxes and compute the location and confidence loss
            match_boxes = gt_boxes[row_mask] # should give a tensor (n_matches, 4)
            match_labels = gt_labels[row_mask] # should give a tensor (n_matches, 1)
                    
            # Loss
            for label, GT_box in zip(match_labels, match_boxes):
                # Confidence loss for the matching category of each GT box
                matches_loss += torch.log(conf_softmax[label])
                # location loss
                loc_loss += smoothL1(pred_box, GT_box)
        else:
            # if it is a negative box, save its greater confidence loss (softmax)
            negatives_losses.append(torch.max(conf_softmax))
        
        
    negative_losses = torch.stack(negative_losses) # make it a torch tensor
    
    # Hard negative mining
    sorted_negatives, indices = torch.sort(negatives, descending=True)
    kept_neg_losses = sorted_negatives[:(hard_mining_ratio*total_matches)]
    nomatch_loss = (torch.log(1+kept_neg_losses)).sum()
    
    # Total confidence loss
    conf_loss = -matches_loss + nomatch_loss
    
    return (conf_loss + loc_loss)/total_matches  