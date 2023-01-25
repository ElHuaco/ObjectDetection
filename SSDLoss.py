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
        self.smoothL1 = nn.SmoothL1Loss(reduction='sum', beta=smoothL1_beta)
        self.hmr = hard_mining_ratio
        self.device = device

    def forward(self, pred_boxes, pred_confidences, gt_boxes, gt_labels, default_boxes):
        # Compute matching between predictions and ground truth boxes

        loss = 0.0
        for b in range(pred_boxes.size(0)):
            matches = matching(default_boxes, gt_boxes[b])  # (N, M) tensor of booleans relating predictions and GT boxes
            # Get loss for every box
            box_matches = matches.sum(dim=1, dtype=torch.bool)
            total_matches = box_matches.sum()
            conf_softmax = pred_confidences[b]
            
            if total_matches > 0:
                # Get the indices of the matches
                indx_matches, indx_matches2 = torch.argwhere(matches), torch.argwhere(matches)

                # Location loss
                coords_matches_gt = torch.index_select(gt_boxes[b], 0, indx_matches[:, 1])
                coords_matches_pred = torch.index_select(pred_boxes[b], 0, indx_matches[:, 0])
                loc_loss = self.smoothL1(coords_matches_pred, coords_matches_gt)

                # Positive confidences loss
                # Gave RunningError for backprop: variable re-declaration is inplace and references itself
                indx_matches2[:,1] = torch.argwhere(torch.index_select(gt_labels[b], 0, indx_matches[:, 1]))[:,1]
                confs_matches_pred = conf_softmax[indx_matches2[:,0], indx_matches2[:,1]]
                matches_loss = torch.sum(torch.log(confs_matches_pred))
            else:
                matches_loss, loc_loss = 0, 0
                total_matches = 1000
                
            # Hard negative mining
            # nope: conf_softmax2 = conf_softmax.clone()
            negative_losses = conf_softmax[torch.logical_not(box_matches)] # da tensor de softmax(confidences) de pred_boxes sin match
            negative_losses = torch.max(negative_losses, dim=1) # coge los maximos solo
            sorted_negatives, indices = torch.sort(negative_losses.values, descending=True)
            try:
                kept_neg_losses = sorted_negatives[:(self.hmr * total_matches)]
            except IndexError:
                kept_neg_losses = sorted_negatives

            nomatch_loss = torch.sum(torch.log(1 + kept_neg_losses))
            #print(f'Confidences loss: {(-matches_loss + nomatch_loss + loc_loss)}')
            #print(f'Localization loss: {loc_loss}')
            loss += (-matches_loss + nomatch_loss + loc_loss) / total_matches

        return loss