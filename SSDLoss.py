# Asumimos que la prediccion es una lista de tensores (c + 4) x DBs x H x W para cada escala
#   Para cada escala: (n_cats + 4) * n_default_boxes * H_scale * W_scale
# La pérdida entonces es, para cada ground truth box de la imagen:
#   i) Las predicciones traducidas a relative-space desde su escala, se matchean con Jaccard a cada ground truth
#   esto se hace con Matching.py
#   ii) Para las que tengan jaccard > 0.5, se calcula L1 de Offset y Softmax de Categorías. Aquí


def confidence_loss(positive_box_conf, negative_box_conf):   
    # Positive examples
    pos_loss = -torch.log(F.softmax(positive_box_conf))
    # Negative examples
    neg_loss = -torch.log(1 - negative_box_conf)
    return (pos_loss + neg_loss)

def location_loss(pred_boxes, gt_boxes, best_gt):    
    return nn.L1Loss()(pred_boxes[best_gt], gt_boxes[best_gt])

# TODO: redefinir SSDLoss como clase de nn.Module
def SSDloss(pred_boxes, pred_confidences, gt_boxes, gt_categories):
    
    # Compute matching between predictions and ground truth boxes
    matches, best_gt = matching(pred_boxes, gt_boxes)
    
    # Perform hard mining
    positive_conf, negative_conf, n_positives = hard_mining(matches, pred_confidences)
    
    #confidence loss
    conf_loss = confidence_loss(positive_conf, negative_conf)
    
    # get pred_boxes[matches] and compare parameters with the associated gt_box
    loc_loss = location_loss(pred_boxes, gt_boxes, best_gt)
    
    loss = (loc_loss + conf_loss)/n_positives # alpha is 1(?)
    
    return loss
    
        