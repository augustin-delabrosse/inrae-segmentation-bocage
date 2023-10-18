import os
import numpy as np
from scipy.signal import convolve2d
import math
import random
from glob import glob
from PIL import Image
import itertools
import tensorflow as tf
from tensorflow import keras
import torch
from torch.nn.functional import interpolate

from keras import backend as K

from glob import glob
import itertools

def get_file_paths(divide_by_dept=False):
    """
    Get file paths for RGB images and mask images.

    Returns:
    - list: List of RGB image paths.
    - list: List of mask image paths.
    """

    if divide_by_dept:
        rgb_img_paths = {}
        masks_img_paths = {}
        depts = [i for i in list(os.walk(r'vignettes/rgb/'))[0][1] if not i.startswith('.')]
        for i in depts:
            rgb_img_paths[f'{i}'] = sorted(
                list(
                    itertools.chain.from_iterable(
                        [glob(i + "*.jpg") for i in glob(f"vignettes/rgb/{i}/*/", recursive=True)]
                        )
                    )
                )
                                
            masks_img_paths[f'{i}'] = sorted(
                list(
                    itertools.chain.from_iterable(
                        [glob(i + "*.png") for i in glob(f"vignettes/mask/{i}/*/", recursive=True)]
                        )
                    )
                )
                                
    else:
        rgb_img_paths = sorted(
            list(
                itertools.chain.from_iterable(
                    [glob(i + "*.jpg") for i in glob("vignettes/rgb/*/*/", recursive=True)]
                )
            )
        )
        masks_img_paths = sorted(
            list(
                itertools.chain.from_iterable(
                    [glob(i + "*.png") for i in glob("vignettes/mask/*/*/", recursive=True)]
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

def get_training_files_paths(input_img_paths, target_img_paths, max_samples, divide_by_dept=True):
    if divide_by_dept:
        if max_samples:
            img_paths = np.array([])
            mask_paths = np.array([])
            
            total_added_samples = 0
            
            for idx, dept in enumerate(input_img_paths.keys()):
                # print(max_samples-total_added_samples, len(input_img_paths.keys())-idx)
                
                n_samples = (max_samples-total_added_samples)/(len(input_img_paths.keys())-idx)
                
                max = len(input_img_paths[dept])
            
                added_samples = int(np.min([n_samples, max]))
                total_added_samples += added_samples
                
                random_indices = get_random_indices(range(added_samples), added_samples)
                random_indices.sort()
            
                img_paths = np.append(img_paths, np.array(input_img_paths[dept])[random_indices])
                mask_paths = np.append(mask_paths, np.array(target_img_paths[dept])[random_indices])
            
                # print(dept, max, n_samples, added_samples)
        else:
            img_paths = sorted({x for v in input_img_paths.values() for x in v})
            mask_paths = sorted({x for v in target_img_paths.values() for x in v})
    else :
        if max_samples:
            random_indices = get_random_indices(range(len(input_img_paths)), max_samples)
            random_indices.sort()
            img_paths = np.array(input_img_paths)[random_indices].tolist()
            mask_paths = np.array(target_img_paths)[random_indices].tolist()
        else:
            img_paths = input_img_paths.copy()
            mask_paths = target_img_paths.copy()

    return img_paths, mask_paths


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


def estimate_noise(img):

  H, W = img.shape

  M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]

  sigma = np.sum(np.sum(np.absolute(convolve2d(img, M))))
  sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

  return sigma


def interpolate_segformer_outputs(preds, output_size: tuple=(256, 256)):
    """
    Bilinear nterpolate segmentation model outputs from a TensorFlow tensor to a PyTorch tensor with a specified output size. 
    The resulting tensor will have a shape of (None, height, width, 1) and can be used for further processing or visualization.

    Args:
        preds (Tensor): A TensorFlow tensor containing segmentation model predictions with shape (None, None, None, 1).
        output_size (tuple, optional): The desired output size of the interpolated tensor in the format (height, width).
            Default is (256, 256).

    Returns:
        Tensor: A PyTorch tensor containing the interpolated segmentation model predictions with shape (None, height, width, 1).

    Example:
        # Resize TensorFlow segmentation model outputs to (256, 256) using bilinear interpolation
        interpolated_preds = interpolate_segformer_outputs(preds, output_size=(256, 256))
    """
    pytorch_preds = torch.from_numpy(preds.numpy()).float()
    
    # Perform interpolation with the correct dimension order and mode
    interpolated_preds = interpolate(pytorch_preds.permute(0, 3, 1, 2), size=output_size, mode='bilinear', align_corners=False)
    
    # Permute dimensions to get the desired shape (1, 256, 256, 1)
    interpolated_preds = interpolated_preds.permute(0, 2, 3, 1)

    return interpolated_preds

def gray_svd_decomposition(img, k):
    img           = Image.fromarray(img)
    img_mat       = np.array(list(img.getdata(band=0)), float)
    img_mat.shape = (img.size[1], img.size[0])
    img_mat       = np.matrix(img_mat)
    
    
    # Perform Singular Value Decomposition
    U, sigma, V = np.linalg.svd(img_mat)
    
    # Image reconstruction
    reconstimg = np.matrix(U[:, :k]) * np.diag(sigma[:k]) * np.matrix(V[:k, :])
    return reconstimg
