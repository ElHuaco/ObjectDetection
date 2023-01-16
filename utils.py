# Use predefined boxes and offsets to get the coordinates of predicted boxes in the input image scale
def offsets2coords(offsets, default_boxes):
    '''
    input: offsets - (B, 4*n_boxes, H, W) - coords in the form (cx, cy, w, h)
    input: default_boxes - (n_boxes, 4) - coords in the form (cx, cy, w, h)
    
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

# Default boxes for training
def create_boxes(aspect_ratios, n_scales=6, scale_min=0.2, scale_max=0.9):
    
    for k in range(1, n_scales+1):
        scale = scale_min + (scale_max - scale_min)*(k - 1)/(n_scales - 1)
        width = [scale*a^(1/2) for a in aspect_ratios]
        height = [scale/a^(1/2) for a in aspect_ratios]
        
        # crear un tensor del tamaño del número de cajas por las dimensiones del feature map
        # y guardar ahí todos los centros con sus respectivos H, W. Repetir para cada escala.
        # Igual es mejor una función para la parte de centros y W, H; porque tenemos varios casos de número de cajas
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    