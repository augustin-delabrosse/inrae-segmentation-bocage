import numpy as np
import cv2
from scipy.ndimage import generic_filter
import os

from tqdm import tqdm

import json
with open('config.json') as f_in:
    config = json.load(f_in)



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

    for idx, path in enumerate(list_of_paths):
        
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

def local_std_dev(image, size=(3, 3, 3)):
    # Define the function to calculate standard deviation for a 3x3 neighborhood
    def std_dev_function(arr):
        return np.std(arr)

    # Apply the standard deviation function to each pixel's 3x3 neighborhood
    result = generic_filter(image, std_dev_function, size=size, output=np.float32, mode='nearest')

    return result


def new_create_large_mask(list_of_paths, predictions, shapes, positions, borders, pred_size=(256, 256), shape_of_the_large_mask=(10000, 10000)):
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

    for idx, path in enumerate(list_of_paths):
        
        basepath = os.path.basename(path)
        
        geo_pos = basepath[9:-4] if len(basepath) < 36 else basepath[15:-4]

        img_pos = positions[geo_pos]
        shape = shapes[geo_pos]
        border = borders[geo_pos]
        top_border = config['border'] if border['top'] else 0
        left_border = config['border'] if border['left'] else 0
        right_border_for_crop = -config['border'] if (border['right'] and shape[0] > pred_size[0]) else None
        right_border_for_pos = config['border'] if (border['right'] and shape[0] > pred_size[0]) else 0
        bottom_border_for_crop = -config['border'] if (border['bottom'] and shape[1] > pred_size[1]) else None
        bottom_border_for_pos = config['border'] if (border['bottom'] and shape[1] > pred_size[1]) else 0
        
        prediction = predictions[idx]
        
        resized_prediction = cv2.resize(prediction, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        
        cropped_prediction = resized_prediction[top_border:bottom_border_for_crop,
                                                left_border:right_border_for_crop]

        mask[round(img_pos[0]*100+top_border):round(img_pos[1]*100-bottom_border_for_pos), round(img_pos[2]*100+left_border): round(img_pos[3]*100-right_border_for_pos)] = cropped_prediction
        
    return mask