# Asumimos que la predicción nos devuelve una lista de tensores (c + 4) x DBs x H x W para cada escala
# 1. Para calcular la loss, separamos los offsets y las confidences, quedándonos con dos listas
# 2. Obtenemos las coordenadas top-left y bottom-right de las pred_boxes utilizando los offsets y la escala
# 3. Relacionamos las ground truth boxes con las predicted boxes; llamamos negativos a todos los pares cuya IoU < 0.5
# 4. Calculamos la loss.

# La loss consta de dos partes:
# - Localization loss: compara las coordenadas de los pares de cajas con relación positiva utilizando una Smooth L1
# - Confidence loss: utiliza cross entropy (log(softmax)) para:
#   - los pares positivos: utilizando la confidence correspondiente a la categoría del match
#   - las cajas de todo pares negativos: utilizando la confidence más alta, con el fin de reducirla.
#    No se utilizarán todas, si no que se escogeran las de mayor pérdida hasta tener un radio de hasta 3:1
#    con los pares positivos

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import matching

class SSDLoss(nn.Module):
    def __init__(self, hard_mining_ratio=3, smoothL1_beta=1.0, device=torch.device('cpu')):
        super(SSDLoss, self).__init__()
        self.smoothL1 = nn.SmoothL1Loss(reduction='mean', beta=smoothL1_beta)
        self.hmr = hard_mining_ratio
        self.device = device

    def forward(self, pred_boxes, pred_confidences, gt_boxes, gt_labels):
        # Compute matching between predictions and ground truth boxes

        matches_loss, nomatch_loss, loc_loss = 0, 0, 0
        for b in range(pred_boxes.size(0)):
            matches = matching(pred_boxes[b], gt_boxes[b])  # (N, M) tensor of booleans relating predictions and GT boxes
            total_matches = matches.sum()
            negative_losses = torch.empty(size=((1,))).to(device=self.device)
            # Get loss for every box
            for i, pred_box in enumerate(pred_boxes[b]):
                # For each predicted box get the matching GTs and their labels
                row_mask = matches[i, :]  # should be a 1-dim tensor of length M - tipo [False, True, True, ...]
                box_matches = row_mask.sum()

                # Softmax of current box confidences
                conf_softmax = F.softmax(pred_confidences[b, i], dim=0)
                # Check if current box matches with a ground truth
                if box_matches != 0:
                    # if it has one or more matches, find to which GT boxes and compute the location and confidence loss
                    match_boxes = gt_boxes[b][row_mask]  # should give a tensor (n_matches, 4)
                    match_labels = torch.argwhere(gt_labels[b][row_mask])[:, 1]  # should give a tensor (n_matches, 1)
                    
                    # Loss
                    for label, GT_box in zip(match_labels, match_boxes):
                        # Confidence loss for the matching category of each GT box
                        matches_loss += torch.log(conf_softmax[label])
                        # location loss
                        loc_loss += self.smoothL1(pred_box, GT_box)
                else:
                    # if it is a negative box, save its greater confidence loss (softmax)
                    negative_losses = torch.cat((negative_losses, torch.max(conf_softmax).unsqueeze(0)))

            # Hard negative mining
            sorted_negatives, indices = torch.sort(negative_losses, descending=True)
            try:
                kept_neg_losses = sorted_negatives[:(self.hmr * total_matches)]
            except IndexError:
                kept_neg_losses = sorted_negatives

            nomatch_loss += (torch.log(1 + kept_neg_losses)).sum()
        
        # Total confidence loss
        conf_loss = -matches_loss + nomatch_loss
        
        return (conf_loss + loc_loss) / total_matches
