import numpy as np
import cv2

import os

from tqdm import tqdm



def create_large_mask(list_of_paths, predictions, shapes, positions, shape_of_the_large_mask=(10000, 10000)):
    """
    Create a large mask for an orthophoto by assembling smaller masks at specified positions.

    Args:
    list_of_paths (list): List of file paths corresponding to smaller masks.
    predictions (list): List of smaller masks (predictions).
    shapes (dict): A dictionary of shapes for the smaller masks.
    positions (dict): A dictionary of positions for the smaller masks.
    shape_of_the_large_mask (tuple): The shape of the large mask.

    Returns:
    numpy.ndarray: A large mask created by assembling smaller masks.
    """
    
    mask = np.zeros(shape_of_the_large_mask)

    for idx, path in tqdm(enumerate(list_of_paths)):
        
        basepath = os.path.basename(path)
        
        geo_pos = basepath[9:-4] if len(basepath) < 36 else basepath[15:-4]

        img_pos = positions[geo_pos]
        shape = shapes[geo_pos]

        mask[round(img_pos[0]*100):round(img_pos[1]*100), round(img_pos[2]*100): round(img_pos[3]*100)] = cv2.resize(predictions[idx], 
                                                                                                             (shape[1], shape[0]), 
                                                                                                             interpolation=cv2.INTER_LINEAR)
        
    return mask

def threshold_array(arr, threshold):
    """
    Apply thresholding to an array. Values below the threshold are set to 0, and values equal to or
    above the threshold are set to 1.

    Args:
    arr (numpy.ndarray): Input array.
    threshold (float): Threshold value.

    Returns:
    numpy.ndarray: Thresholded array..
    """
    # Create a copy of the input array to avoid modifying the original
    result = arr.copy()
    
    # Apply the thresholding operation
    result[result < threshold] = 0
    result[result >= threshold] = 1
    
    return result