import numpy as np
import random
from glob import glob
import itertools
import tensorflow as tf
from tensorflow import keras

from keras import backend as K

def get_training_file_paths():
    """
    Get file paths for RGB images and mask images.

    Returns:
    - list: List of RGB image paths.
    - list: List of mask image paths.
    """
    # Define your dataset paths
    rgb_img_paths = sorted(
        list(
            itertools.chain.from_iterable(
                [glob(i + "*.jpg") for i in glob("vignettes/rgb/*/", recursive=True)]
            )
        )
    )
    masks_img_paths = sorted(
        list(
            itertools.chain.from_iterable(
                [glob(i + "*.png") for i in glob("vignettes/mask/*/", recursive=True)]
            )
        )
    )
    return rgb_img_paths, masks_img_paths


def threshold_array(arr, threshold):
    # Create a copy of the input array to avoid modifying the original
    result = arr.copy()
    
    # Apply the thresholding operation
    result[result < threshold] = 0
    result[result >= threshold] = 1
    
    return result


def get_class_weights(masks):
    ones = 0
    for mask in tqdm(masks):
        shape = mask.shape
        
        if mask[:,:,0].max() > 1.:
            ones += (mask/255.).sum()
        else:
            ones += (mask).sum()
    
    total = len(masks) * shape[0] * shape[1]
    weight_of_ones = ones/total
    weight_of_zeros = 1 - weight_of_ones
    return {0: round(weight_of_zeros, 3), 0: round(weight_of_ones, 3)}


def get_random_indices(input_list, n):
    """
    Get a list of n random indices from the input list.

    Args:
    input_list (list): The list of elements from which to select random indices.
    n (int): The number of random indices to select.

    Returns:
    list: A list of n random indices.
    """
    if n > len(input_list):
        raise ValueError("n cannot be greater than the length of the input list")

    # Use random.sample to select n unique random indices
    random_indices = random.sample(range(len(input_list)), n)
    
    return random_indices


def load_custom_model(model_path, custom_objects_list):
    custom_objects_dict = {}
    for i in custom_objects_list:
        if 'function ' in str(i):
            custom_objects_dict[str(i).split(' at ')[0].split('.')[-1]] = i
        elif 'bound method ' in str(i):
            custom_objects_dict[str(i).split(' of ')[0].split('.')[-1]] = i
    loaded_model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects_dict
    )
    return loaded_model