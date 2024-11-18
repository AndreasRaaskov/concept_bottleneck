"""
Contains functions related to calculating saliency maps and scores for parts of an image.
And to get the saliency score for visible consepts

"""

import torch
import numpy as np

from captum.attr import NoiseTunnel, Saliency, LayerGradCam


def get_saliency_maps(img, target_classes,model, method_type='vanilla'):
    """
    Function to compute saliency maps for a given image and a list of classes.

    Args:
        img (torch.Tensor): Input image tensor of shape (C, H, W).
        target_classes (List[int]): List of target classes for which to compute saliency maps.
        model (torch.nn.Module): The model for which to compute saliency maps.
        method_type (str): The method to use for computing saliency maps. Options: 'vanilla', 'noise_tunnel', 'gradcam'.
                           Default is 'vanilla'.
    """

    # Load and preprocess the image
    img.requires_grad_(True)

    # Initialize the appropriate method
    if method_type == 'vanilla':
        saliency = Saliency(model)
    elif method_type == 'noise_tunnel':
        saliency = NoiseTunnel(Saliency(model))
    elif method_type == 'gradcam':
        layer = model.model.Conv2d_2b_3x3
        saliency = LayerGradCam(model, layer)
    else:
        raise ValueError("Invalid method_type. Choose from 'vanilla', 'noise_tunnel', or 'gradcam'.")

    
    
    # Create a batch by repeating the input image
    batched_input = img.repeat(len(target_classes), 1, 1, 1)

    if method_type == 'vanilla':
        # Compute vanilla saliency
        attribution = saliency.attribute(batched_input, target=target_classes,abs=True)
    """
    # Loop over all the target classes
    for target_class in target_classes:
        if method_type == 'vanilla':
            # Compute vanilla saliency
            attribution = saliency.attribute(img, target=int(target_class))
        elif method_type == 'noise_tunnel':
            # Compute saliency with noise tunneling
            attribution = saliency.attribute(img, target=int(target_class), nt_type='smoothgrad', nt_samples=50, stdevs=0.2)
    """ 
    """
    TODO fingure out how to get gradcam to work with inception
    elif method_type == 'gradcam':
        # Compute GradCAM
        attribution = saliency.attribute(img, target=target_class)
        attribution = torch.mean(attribution, dim=1, keepdim=True)  # Average across channels
    """
    saliency_maps = []
    for map in attribution:
        map = map.squeeze().cpu().detach().numpy().sum(axis=0)  # Convert attribution to numpy array and sum across channels
        if np.max(map) != 0:
            map = map / np.sum(map)  # Normalize to sum to 1        
            saliency_maps.append(map)  


    return saliency_maps

def saliency_score_part(saliency_map, coordinates):
    """
    Calculate a score of a single part based on the sailency activation times the manhatten distance to the coordinates. 
    args: sailency: np.array: The sailency map
    cords: list of tuples: The coordinates to calculate the distance to
    return: float: The score
    """

    height, width = saliency_map.shape  # Get the dimensions of the saliency map

    # Create coordinate grids for the entire map
    y_coords, x_coords = np.ogrid[:height, :width]

    distances = []
    
    for target_y, target_x in coordinates:
        # Calculate Manhattan distances from each point to the target coordinate
        distances.append(np.abs(y_coords - target_y) + np.abs(x_coords - target_x))
    
    if len(distances) ==1: #If only one coordinate return the sum of the score
        M = distances[0]

    else:
        # If multiple coordinates return the minimum distance for each coordinate
        M = np.stack(distances).min(axis=0)

    #Return the score
    return (saliency_map * M).sum()/M.mean()

def saliency_score_image(sailency, coordinates):
    """
    Calculate a average score of all the parts in the image
    args: sailency: list of np.array: The sailency maps
    cords: list of list of tuples: The coordinates to calculate the distance to
    return: float: The score
    """

    score = 0
    for s,c in zip(sailency,coordinates):
        score += saliency_score_part(s,c)
    
    return score/len(coordinates)

def get_visible_consepts(coordinates):
    """
    Get the indexes of the visible consepts
    args: coordinates: list of list of tuples: The coordinates to calculate the distance to
    return: list of int: The indexes of the visible consepts
    """
    # Find parts that are in the image, a part is in the image if it has at least one coordinate
    visible_idx = []
    visible_consepts = []
    for i,c in enumerate(coordinates):
        if len(c) > 0: #Check if the part has any coordinates
            if c[0] != (0,0): #Check if the coordinates are not default for not visible
                visible_idx.append(i)
                visible_consepts.append(c)
    return visible_idx,visible_consepts