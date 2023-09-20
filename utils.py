import numpy as np
import random

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