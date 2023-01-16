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


# Default boxes for training
def create_boxes(aspect_ratios, scale, FM_size, extra_box_scale=None):
    '''
    input: aspect_ratios - set of aspect ratios for the default boxes
    input: scale - scale of the default boxes for the specific feature map
    input: FM_size - size of the specific feature map
    
    output: default_boxes - tensor (n_boxes*FM_size**2, 4) with the default boxes for the specific feature map and scale
    '''
    # Build tensor for storing all the boxes coordinates
    n_boxes = len(aspect_ratios)
    total_boxes = FM_size**2*n_boxes
    default_boxes = torch.empty(size=(total_boxes, 4))
    
    # Widths and heights of the default boxes
    widths = [scale*a**(1/2) for a in aspect_ratios]
    heights = [scale/a**(1/2) for a in aspect_ratios]
    if extra_box_scale is not None:
        widths.append(extra_box_scale*a**(1/2))
        heights.append(extra_box_scale/a**(1/2))
    
    prev_indx, i, j = 0, 0, 0
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