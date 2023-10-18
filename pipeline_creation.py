import cv2
import tensorflow as tf
from glob import glob
import itertools
import numpy as np
import random
from tqdm import tqdm

from utils import get_file_paths

AUTOTUNE = tf.data.experimental.AUTOTUNE


def weights_to_apply():
    rgb_img_paths, masks_img_paths = get_file_paths()
    ones = 0
    for i in tqdm(masks_img_paths):
        mask = cv2.imread(i)
        shape = mask[:,:,0].shape
        
        if mask[:,:,0].max() > 1.:
            ones += (mask[:,:,0]/255.).sum()
        else:
            ones += (mask[:,:,0]).sum()
    
    total = len(masks_img_paths) * shape[0] * shape[1]
    zeros = total - ones
    
    weight_for_0 = (1 / zeros) * (total / 2.0)
    weight_for_1 = (1 / ones) * (total / 2.0)
    
    weights_to_apply = {0: weight_for_0, 1:weight_for_1}

    return weights_to_apply

# Function to preprocess an image with histogram equalization
def preprocess_image(image_path, mask_path):
    """
    Preprocess an RGB image and its mask by applying histogram equalization and normalization.

    Parameters:
    - image_path (str): Path to the RGB image.
    - mask_path (str): Path to the mask image.

    Returns:
    - tuple: A tuple containing the preprocessed image and mask.
    """
    # RGB image
    # Read an image from a file
    image_string = tf.io.read_file(image_path)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    # Apply histogram equalization
    image_equalized = tf.numpy_function(equalize_histogram, [image_decoded], tf.float32)
    # Normalize the image
    image_normalized = image_equalized/255.# (image_equalized - tf.reduce_min(image_equalized)) / (tf.reduce_max(image_equalized) - tf.reduce_min(image_equalized))
    # To grayscale
    # image_grayscaled = tf.image.rgb_to_grayscale(image_normalized)
    # Crop the image
    image_output = tf.image.crop_to_bounding_box(
        image_normalized, 0, 0, 256, 256
    )

    # Mask (same steps except no equalizing and normalizing)
    mask_string = tf.io.read_file(mask_path)
    mask_decoded = tf.image.decode_jpeg(mask_string, channels=1)
    mask_output = tf.image.crop_to_bounding_box(
        mask_decoded, 0, 0, 256, 256
    )
    # Ensure mask_output is of type float32
    mask_output = tf.cast(mask_output, tf.float32)/255.

    return image_output, mask_output

# Function to apply histogram equalization to an image
def equalize_histogram(image, convert_to_tensors=True):
    """
    Apply histogram equalization to an RGB image.

    Parameters:
    - image (tf.Tensor): Input RGB image.

    Returns:
    - tf.Tensor: Image after histogram equalization.
    """
    # Convert if needed to np.uint8 format
    # if np.max(image) <= 1.:
    #     image = np.uint8(image*255)
    
    # Convert RGB to YUV
    image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    # Apply histogram equalization to the Y channel
    image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
    # Convert YUV back to RGB
    image_output = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

    if convert_to_tensors:
        # Convert back to tensor
        image_output = tf.convert_to_tensor(image_output, dtype=tf.float32)
    return image_output

def gaussian_blur(image, kernel_size=(3, 3)):
    """
    Apply Gaussian blur to an image.

    Parameters:
    - image (tf.Tensor): Input image.
    - kernel_size (tuple): Size of the Gaussian kernel.

    Returns:
    - tf.Tensor: Image after Gaussian blur.
    """
    image = cv2.GaussianBlur(image, kernel_size, 0)
    return image

def data_augmentation(image, mask, grayscale=False):
    """
    Apply data augmentation techniques to an image and its mask.

    Parameters:
    - image (tf.Tensor): Input image.
    - mask (tf.Tensor): Input mask.

    Returns:
    - tuple: A tuple containing the augmented image and mask.
    """
    # Randomly flip left-right
    if tf.random.uniform(()) > 0.25:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # Randomly flip up-down
    if tf.random.uniform(()) > 0.25:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    # Apply color transformation (RGB to grayscale)
    if grayscale:
        image = tf.image.rgb_to_grayscale(image)

    # Apply Gaussian blurring
    if tf.random.uniform(()) > 0.25:
        image = tf.numpy_function(gaussian_blur, [image], tf.float32)
        image = tf.convert_to_tensor(image, dtype=tf.float32)

    return image, mask

def create_pipeline(train_samples, val_samples, data_augment=True):
    """
    Create a TensorFlow data pipeline for image preprocessing and augmentation.

    Returns:
    - tf.data.Dataset: Combined dataset with preprocessed and augmented images.
    """
    # Get paths for both RGB images and masks
    rgb_img_paths, masks_img_paths = get_file_paths()

    # Shuffle
    c = list(zip(rgb_img_paths, masks_img_paths))
    random.shuffle(c)
    rgb_img_paths, masks_img_paths = zip(*c)
    
    train_rgb_img_paths = list(rgb_img_paths[:train_samples])
    train_masks_img_paths = list(masks_img_paths[:train_samples])

    val_rgb_img_paths = list(rgb_img_paths[train_samples:train_samples+val_samples])
    val_masks_img_paths = list(masks_img_paths[train_samples:train_samples+val_samples])
    # return train_rgb_img_paths, train_masks_img_paths, val_rgb_img_paths, val_masks_img_paths
    # Create a first dataset of image and mask paths
    train_ds = tf.data.Dataset.from_tensor_slices((train_rgb_img_paths, train_masks_img_paths))
    val_ds = tf.data.Dataset.from_tensor_slices((val_rgb_img_paths, val_masks_img_paths))

    # Apply preprocessing steps to the images and the masks
    train_ds = train_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)

    if data_augment:
        # Apply data augmentation to the dataset
        augmented_train_dataset = train_ds.map(data_augmentation, num_parallel_calls=AUTOTUNE)
    
        # Concatenate the original and the new datasets
        train_ds = train_ds.concatenate(augmented_train_dataset)

    return train_ds, val_ds, train_rgb_img_paths, train_masks_img_paths, val_rgb_img_paths, val_masks_img_paths
    

def make_list_of_elements(dataset):
    """
    Convert a TensorFlow dataset into a list of elements.

    Parameters:
    - dataset (tf.data.Dataset): Input TensorFlow dataset.

    Returns:
    - list: List of elements from the dataset.
    """
    elems = list(dataset.as_numpy_iterator())
    return elems
