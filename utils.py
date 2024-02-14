import re
from tqdm import tqdm
import os
import numpy as np
from scipy.signal import convolve2d
import math
import random
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf
from tensorflow import keras
# import torch
# from torch.nn.functional import interpolate

# from keras import backend as K

def get_file_paths(divide_by_dept=False, large=False, year=2020):
    """
    Get file paths for RGB images and mask images.

    Returns:
    - list: List of RGB image paths.
    - list: List of mask image paths.
    """
    
    
    
    if year == 2020:
        path_img = 'vignettes/rgb/'
        path_mask = 'vignettes/mask/'
    else:
        path_mask = 'lorem/'
        if large:
            path_img = f'vignettes/rgb_older/rgb_{str(year)}_large/'
        else:
            path_img = f'vignettes/rgb_older/rgb_{str(year)}/'
            
    
    if large:
        if divide_by_dept:
            rgb_img_paths = {}
            masks_img_paths = {}
            depts = [i for i in list(os.walk(f'{path_img}'))[0][1] if not i.startswith('.') and i.endswith('large')]
            
            for i in depts:
                rgb_img_paths[f'{i}'] = sorted(
                    list(
                        itertools.chain.from_iterable(
                            [glob(i + "*.jpg") for i in glob(f"{path_img}{i}/*/", recursive=True)]
                            )
                        )
                    )

                masks_img_paths[f'{i}'] = sorted(
                    list(
                        itertools.chain.from_iterable(
                            [glob(i + "*.png") for i in glob(f"{path_mask}{i}/*/", recursive=True)]
                            )
                        )
                    )

        else:
            rgb_img_paths = sorted(
                list(
                    itertools.chain.from_iterable(
                        [glob(i + "*.jpg") for i in glob(f"{path_img}*_large/*/", recursive=True)]
                    )
                )
            )
            masks_img_paths = sorted(
                list(
                    itertools.chain.from_iterable(
                        [glob(i + "*.png") for i in glob(f"{path_mask}*_large/*/", recursive=True)]
                    )
                )
            )
    else:
        if divide_by_dept:
            rgb_img_paths = {}
            masks_img_paths = {}
            depts = [i for i in list(os.walk(f'{path_img}'))[0][1] if not i.startswith('.') and not i.endswith('large')]

            for i in depts:
                rgb_img_paths[f'{i}'] = sorted(
                    list(
                        itertools.chain.from_iterable(
                            [glob(i + "*.jpg") for i in glob(f"{path_img}{i}/*/", recursive=True)]
                            )
                        )
                    )

                masks_img_paths[f'{i}'] = sorted(
                    list(
                        itertools.chain.from_iterable(
                            [glob(i + "*.png") for i in glob(f"{path_mask}{i}/*/", recursive=True)]
                            )
                        )
                    )

        else:
            rgb_img_paths = sorted(
                list(
                    itertools.chain.from_iterable(
                        [glob(i + "*.jpg") for i in glob(f"{path_img}*/*/", recursive=True)]
                    )
                )
            )
            masks_img_paths = sorted(
                list(
                    itertools.chain.from_iterable(
                        [glob(i + "*.png") for i in glob(f"{path_mask}*/*/", recursive=True)]
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


def get_random_indices(input_list, n, seed=101):
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
    random.seed(seed)
    random_indices = random.sample(range(len(input_list)), n)
    
    return random_indices

def shuffle_lists_of_img_and_masks(img_list, img_mask):
    indices = np.arange(len(img_list))
    np.random.shuffle(indices)
    
    img_list = list(np.array(img_list)[indices])
    img_mask = list(np.array(img_mask)[indices])

    return img_list, img_mask

def get_training_files_paths(input_img_paths, target_img_paths, max_samples, divide_by_dept=True, year=2020):
    """
    Get training file paths for input and target images. If 'divide_by_dept'
    is True, the function returns an equal number of file paths for each 
    administrative department, ensuring a balanced distribution. 
    If 'max_samples' is set, it limits the number of samples retrieved.

    Args:
    input_img_paths (dict or list): A dictionary mapping department keys to lists of input image file paths,
                                    or a list of input image file paths.
    target_img_paths (dict or list): A dictionary mapping department keys to lists of target image file paths,
                                     or a list of target image file paths.
    max_samples (int): The maximum number of samples to retrieve.
    divide_by_dept (bool): If True, divide samples by department; otherwise, combine all samples.

    Returns:
    tuple: A tuple containing two lists, the first for input image file paths and the second for target image file paths
    """

    if divide_by_dept:
        if max_samples:
            img_paths = np.array([])
            mask_paths = np.array([])
            total_added_samples = 0

            for idx, dept in enumerate(input_img_paths.keys()):
                # Calculate the number of samples to add per department
                n_samples = (max_samples - total_added_samples) / (len(input_img_paths.keys()) - idx)
                max = len(input_img_paths[dept])

                # Calculate the number of added samples for this department
                added_samples = int(np.min([n_samples, max]))
                total_added_samples += added_samples

                # Get random indices for selecting samples
                random_indices = get_random_indices(range(added_samples), added_samples)
                random_indices.sort()

                img_paths = np.append(img_paths, np.array(input_img_paths[dept])[random_indices])
                if year == 2020:
                    mask_paths = np.append(mask_paths, np.array(target_img_paths[dept])[random_indices])
        else:
            # Combine all input and target image paths
            img_paths = sorted({x for v in input_img_paths.values() for x in v})
            if year == 2020:
                mask_paths = sorted({x for v in target_img_paths.values() for x in v})
    else:
        if max_samples:
            # Randomly select samples when not dividing by department
            random_indices = get_random_indices(range(len(input_img_paths)), max_samples)
            random_indices.sort()
            img_paths = np.array(input_img_paths)[random_indices].tolist()
            if year == 2020:
                mask_paths = np.array(target_img_paths)[random_indices].tolist()
        else:
            # Copy input and target image paths as is
            img_paths = input_img_paths.copy()
            if year == 2020:
                mask_paths = target_img_paths.copy()

    if year != 2020:
        mask_paths = []
        
    return img_paths, mask_paths



def load_custom_model(model_path, custom_objects_list):
    """
    Load a custom model from a file with the provided custom objects.

    Args:
    model_path (str): The file path to the saved model.
    custom_objects_list (list): A list of custom objects used in the model.

    Returns:
    tf.keras.Model: The loaded model with custom objects.

    This function loads a custom model from a file and ensures that the custom objects used in the model are correctly
    loaded by creating a dictionary mapping the custom object names to the objects themselves. It then loads the model
    using TensorFlow's 'tf.keras.models.load_model' method with the custom objects dictionary.
    """
    custom_objects_dict = {}
    for i in custom_objects_list:
        if 'function ' in str(i):
            custom_objects_dict[str(i).split(' at ')[0].split('.')[-1]] = i
        elif 'bound method ' in str(i):
            custom_objects_dict[str(i).split(' of ')[0].split('.')[-1]] = i
        else:
            custom_objects_dict[str(i).split('.')[-1][:-2]] = i
    loaded_model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects_dict
    )
    
    return loaded_model

def estimate_noise(img):
    """
    Estimate the noise level in an image using a simple algorithm inspired by John Immerkær's paper "Fast Noise Variance Estimation."
    It calculates the noise by applying a convolution operation to the image with a
    specific 3x3 filter and then computing the noise level based on the resulting values.
    
    Args:
    img (numpy.ndarray): Input image for which noise level needs to be estimated.

    Returns:
    float: Estimated noise level. 
    """
    
    H, W = img.shape

    M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(img, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

    return sigma


# def interpolate_segformer_outputs(preds, output_size: tuple=(256, 256)):
#     """
#     Bilinear nterpolate segmentation model outputs from a TensorFlow tensor to a PyTorch tensor with a specified output size. 
#     The resulting tensor will have a shape of (None, height, width, 1) and can be used for further processing or visualization.

#     Args:
#         preds (Tensor): A TensorFlow tensor containing segmentation model predictions with shape (None, None, None, 1).
#         output_size (tuple, optional): The desired output size of the interpolated tensor in the format (height, width).
#             Default is (256, 256).
# 
#     Returns:
#         Tensor: A PyTorch tensor containing the interpolated segmentation model predictions with shape (None, height, width, 1).

#     Example:
#         # Resize TensorFlow segmentation model outputs to (256, 256) using bilinear interpolation
#         interpolated_preds = interpolate_segformer_outputs(preds, output_size=(256, 256))
#     """
#     pytorch_preds = torch.from_numpy(preds.numpy()).float()
    
    # Perform interpolation with the correct dimension order and mode
#     interpolated_preds = interpolate(pytorch_preds.permute(0, 3, 1, 2), size=output_size, mode='bilinear', align_corners=False)
    
    # Permute dimensions to get the desired shape (1, 256, 256, 1)
#     interpolated_preds = interpolated_preds.permute(0, 2, 3, 1)

#     return interpolated_preds

def gray_svd_decomposition(img, k):
    """
    Perform Singular Value Decomposition (SVD) on a grayscale image and reconstruct it 
    by performing Singular Value Decomposition (SVD) using the top 'k' singular values.

    Args:
    img (numpy.ndarray): Grayscale image as a NumPy array.
    k (int): The number of singular values to retain for reconstruction.

    Returns:
    numpy.ndarray: Reconstructed grayscale image.
    """
    # Convert the NumPy array to a PIL Image
    img = Image.fromarray(img)
    
    # Convert the image data to a NumPy matrix
    img_mat = np.array(list(img.getdata(band=0), dtype=float))
    img_mat.shape = (img.size[1], img.size[0])
    img_mat = np.matrix(img_mat)
    
    # Perform Singular Value Decomposition
    U, sigma, V = np.linalg.svd(img_mat)
    
    # Reconstruct the image using the top 'k' singular values
    reconstimg = np.matrix(U[:, :k]) * np.diag(sigma[:k]) * np.matrix(V[:k, :])
    
    return reconstimg


import numpy as np

def svd_compressor(image, k):
    """
    Compress a 1D image using Singular Value Decomposition (SVD) by retaining only the top 'k' singular values.

    Args:
    image (numpy.ndarray): Input 1D image as a NumPy array.
    k (int): The number of singular values to retain for compression.

    Returns:
    numpy.ndarray: Compressed image.
    """
    # Create an array filled with zeros having the shape of the input image
    compressed = np.zeros(image.shape)
    
    # Get the U, S, and V terms (S = SIGMA)
    U, S, V = np.linalg.svd(image)
    
    # Loop over U columns (Ui), S diagonal terms (Si), and V rows (Vi) until the chosen order
    for i in range(k):
        Ui = U[:, i].reshape(-1, 1)
        Vi = V[i, :].reshape(1, -1)
        Si = S[i]
        compressed += Ui * Si * Vi
    
    return compressed

def rgb_svd_decomposition(image, k):
    """
    Perform RGB image decomposition using Singular Value Decomposition (SVD) on each color channel.

    Args:
    image (numpy.ndarray): Input RGB image as a NumPy array.
    k (int): The number of singular values to retain for decomposition.

    Returns:
    numpy.ndarray: Reconstructed RGB image.

    This function separates an RGB image into its red, green, and blue color channels. It then applies the 'svd_compressor'
    function to each color channel, retaining only the top 'k' singular values. Finally, it combines the compressed color
    channels to reconstruct the RGB image.
    """
    # Separation of the image channels
    red_image = np.array(image)[:, :, 0]
    green_image = np.array(image)[:, :, 1]
    blue_image = np.array(image)[:, :, 2]

    # Compress each color channel using SVD
    red_comp = svd_compressor(red_image, k)
    green_comp = svd_compressor(green_image, k)
    blue_comp = svd_compressor(blue_image, k)

    # Combine the compressed color channels
    color_comp = np.zeros((np.array(image).shape[0], np.array(image).shape[1], 3))
    color_comp[:, :, 0] = red_comp
    color_comp[:, :, 1] = green_comp
    color_comp[:, :, 2] = blue_comp
    color_comp = np.around(color_comp).astype(int)
    
    return color_comp

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def get_commun_coordinates_paths(rgb_dir, mask_dir, dept):
    """
    Get common coordinates and paths between RGB and mask images.

    Args:
        rgb_dir (str): Directory containing RGB images.
        mask_dir (str): Directory containing mask images.
        dept (str): Department code.

    Returns:
        tuple: Tuple of lists containing common RGB and mask image paths.

    This function retrieves common coordinates and paths between RGB and mask images in specified directories for a given department.
    """
    # Extract the year from the RGB directory
    m_year = int(re.findall(r'[0-9]+', mask_dir)[0])
    m_dir = mask_dir.split('/')[0]

    # Get paths of all RGB and mask images
    input_img_paths = sorted(list(itertools.chain.from_iterable([glob(i + "*.jpg") for i in glob(rgb_dir, recursive=True)])))
    target_img_paths = sorted(list(itertools.chain.from_iterable([glob(i + "*.png") for i in glob(mask_dir, recursive=True)])))

    # Extract coordinates from each element in the list
    get_coordinates = lambda x: tuple(map(int, os.path.splitext(os.path.basename(x))[0].split('_')[-2:]))

    # Get coordinates from both lists
    coordinates_target_img_paths = set(get_coordinates(item) for item in target_img_paths)
    coordinates_input_img_paths = set(get_coordinates(item) for item in input_img_paths)

    # Find common coordinates
    common_coordinates = coordinates_target_img_paths.intersection(coordinates_input_img_paths)

    # Filter paths based on common coordinates
    result_img = [item for item in tqdm(input_img_paths) if get_coordinates(item) in common_coordinates]

    # Generate corresponding mask paths
    coordinates = [get_coordinates(i) for i in result_img]
    result_mask = [f"{m_dir}/{m_year}\\{dept}\\pred_{m_year}_{dept}_{i[0]}_{i[1]}.png" for i in coordinates]


    return result_img, result_mask


def remove_empty_vignettes(rgb_paths_list, mask_paths_list):
    """
    Remove empty vignettes from lists of RGB and mask paths.

    This function iterates through the RGB images, checks if each image is mostly white (empty),
    and removes corresponding entries from both RGB and mask paths lists.

    Args:
        rgb_paths_list (list): List of paths to RGB images.
        mask_paths_list (list): List of paths to mask images corresponding to RGB images.

    Returns:
        tuple: Updated lists of RGB and mask paths after removing empty vignettes.

    Example:
        rgb_paths, mask_paths = remove_empty_vignettes(rgb_paths, mask_paths)
    """
    # Initialize a counter for empty vignettes
    num = 0
    
    # Create copies of the input lists to avoid modifying them directly
    copy_rgb_paths_list = rgb_paths_list.copy()
    copy_mask_paths_list = mask_paths_list.copy()
    
    # Iterate through the copied list of RGB paths with tqdm for progress tracking
    for idx, path in tqdm(enumerate(copy_rgb_paths_list)):
        # Read the image using Matplotlib's imread function
        img = plt.imread(path)
        
        # Check if the minimum pixel value of the image is greater than 240 (mostly white)
        if img.min() > 240:
            # Remove the corresponding entries from both RGB and mask paths lists
            rgb_paths_list.remove(copy_rgb_paths_list[idx])
            mask_paths_list.remove(copy_mask_paths_list[idx])
            # Increment the counter for empty vignettes
            num += 1
    
    # Calculate the percentage of empty vignettes and print the result
    empty_percentage = round(num / len(copy_rgb_paths_list), 4) * 100
    print(f"{empty_percentage}% of the vignettes are empty.")
    
    # Return the updated lists of RGB and mask paths
    return rgb_paths_list, mask_paths_list


def calculate_mean_std(image):
    """
    Calculate the mean and standard deviation values for each channel of a numpy image array.

    Parameters:
    - image (numpy.ndarray): The input image array with shape (height, width, channels).

    Returns:
    - mean_values (numpy.ndarray): An array containing the mean values for each channel.
    - std_values (numpy.ndarray): An array containing the standard deviation values for each channel.
    
    Example:
    >>> import numpy as np
    >>> your_image_array = np.random.rand(100, 100, 3)  # Replace with actual image array
    >>> mean_values, std_values = calculate_mean_std(your_image_array)
    >>> print("Mean values for each channel:", mean_values)
    >>> print("Standard deviation values for each channel:", std_values)
    """
    # Calculate mean values for each channel
    mean_values = np.mean(image, axis=(0, 1))

    # Calculate standard deviation values for each channel
    std_values = np.std(image, axis=(0, 1))

    return mean_values, std_values

import cv2  # You may need to install OpenCV if not already installed
import numpy as np
from time import time

def calculate_global_mean_std(image_paths):
    """
    Calculate the global mean and standard deviation values for a list of images.

    Parameters:
    - image_paths (list): A list of file paths to the images.

    Returns:
    - global_mean (numpy.ndarray): An array containing the global mean values across all images and channels.
    - global_std (numpy.ndarray): An array containing the global standard deviation values across all images and channels.
    
    Example:
    >>> image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
    >>> global_mean, global_std = calculate_global_mean_std(image_paths)
    >>> print("Global Mean values:", global_mean)
    >>> print("Global Standard deviation values:", global_std)
    """

    start = time()
    # Initialize variables to accumulate mean and std values
    total_mean = np.zeros(3)
    total_std = np.zeros(3)

    for image_path in image_paths:
        # Read the image using OpenCV
        image = cv2.imread(image_path)
        
        # Calculate mean and std for the current image
        mean_values, std_values = calculate_mean_std(image)

        # Accumulate mean and std values
        total_mean += mean_values
        total_std += std_values

    # Calculate global mean and std across all images
    num_images = len(image_paths)
    global_mean = total_mean / num_images
    global_std = total_std / num_images

    print(f'computations performed  in {round(time()-start, 2)} sec')
    return global_mean, global_std