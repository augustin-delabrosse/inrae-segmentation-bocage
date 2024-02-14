from utils import get_random_indices, get_file_paths, gray_svd_decomposition, rgb_svd_decomposition# , hist_match

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.image import per_image_standardization
from sklearn.model_selection import train_test_split


from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2 
from skimage.restoration import denoise_wavelet# , denoise_tv_bregman, denoise_tv_chambolle
from skimage.util import random_noise
import re
    
from PIL import Image, ImageOps, ImageFilter
    
# from PIL import Image

import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from scipy.ndimage import binary_erosion

import warnings
warnings.simplefilter("ignore")


# from keras import backend as K

# K.set_image_data_format('channels_last')  # Set the data format to 'channels_last' (TensorFlow dimension ordering)


class Augmentation:
    def __init__(self):
        self.seq = iaa.Sequential([
            # iaa.Crop(percent=(0, 0.4), keep_size=True),
            # iaa.Sometimes(0.5, iaa.CropAndPad(percent=(-0.4, 0.4))),
            # iaa.MultiplyBrightness((0.5, 1.5)),
            iaa.Sometimes(0.5, iaa.CoarseDropout(0.1, size_percent=1)),
            # iaa.Sometimes(0.5, iaa.Affine(rotate=(-30, 30))),
            iaa.Sometimes(0.5, iaa.Fliplr(0.5)),
            # iaa.Sometimes(0.5, iaa.Flipud(0.5)),
            # iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 0.7))),
            iaa.Sometimes(0.5, iaa.ElasticTransformation(alpha=10, sigma=1)),
            # iaa.Sometimes(1, iaa.Cutout(nb_iterations=(1, 3), size=0.2, squared=False))
        ])

    @staticmethod
    def crop(img, mask, size):
        img = np.array(img)
        mask = np.array(mask)
        w, h, _ = img.shape
        padw = size - w if w < size else 0
        padh = size - h if h < size else 0
        img = np.pad(img, [(0, padw), (0, padh), (0, 0)], mode='constant', constant_values=0)
        mask = np.pad(mask, [(0, padw), (0, padh)], mode='constant', constant_values=0)
    
        w, h, _ = img.shape
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        img = img[x:x + size, y:y + size, :]
        mask = mask[x:x + size, y:y + size]
        return img, mask

    @staticmethod
    def hflip(img, mask, p=0.5):
        img = np.array(img)
        mask = np.array(mask)
        if random.random() < p:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        return img, mask

    @staticmethod
    def normalize(img, mask=None, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        img = (img / 255.0 - mean) / std
        img = tf.convert_to_tensor(img)
        if mask is not None:
            mask = tf.convert_to_tensor(mask, dtype=tf.int32)
            return img, mask
        return img

    @staticmethod
    def resize(img, mask, base_size, ratio_range):
        """
        Normalize the input image and mask to torch tensors.
    
        Args:
            img (PIL Image): Input image.
            mask (PIL Image): Corresponding mask.
    
        Returns:
            torch.Tensor, torch.Tensor: Normalized image and mask tensors.
        """
        w, h, _ = img.shape
        long_side = random.randint(int(base_size * ratio_range[0]), int(base_size * ratio_range[1]))
    
        if h > w:
            oh = long_side
            ow = int(1.0 * w * long_side / h + 0.5)
        else:
            ow = long_side
            oh = int(1.0 * h * long_side / w + 0.5)

        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
        return img, mask 

    @staticmethod
    def colorjitter(img, parameters=(0.5, 0.5, 0.5, 0.2), p=0.8):
        if random.random() < p:
            img = tf.image.random_hue(img, max_delta=parameters[0])
            img = tf.image.random_saturation(img, lower=1-parameters[1], upper=1+parameters[1])
            img = tf.image.random_contrast(img, lower=1-parameters[2], upper=1+parameters[2])
            img = tf.image.random_brightness(img, max_delta=parameters[3]).numpy()
        return img

    @staticmethod
    def to_grayscale_converter(img, p=0.2):
        if random.random() < p:
            if str(type(img)).split("'")[1].split('.')[0] != 'PIL':
                img = Image.fromarray(img)
            img_ = np.array(img.convert('L'))

            img = np.zeros(img_.shape + (3,))
            img[:,:,0] = img_
            img[:,:,1] = img_
            img[:,:,2] = img_
        return img

    @staticmethod
    def blur(img, p=0.5):
        if str(type(img)).split("'")[1].split('.')[0] != 'PIL':
            img = Image.fromarray(img.astype('uint8'))
        if random.random() < p:
            sigma = np.random.uniform(0.1, 2.0)
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return np.array(img)

    @staticmethod
    def cutout(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3,
               ratio_2=1/0.3, value_min=0, value_max=255, pixel_level=True):
        """
        Apply cutout to the input image and mask with a given probability.
    
        Args:
            img (PIL Image): Input image.
            mask (PIL Image): Corresponding mask.
            p (float): Probability of applying cutout.
            size_min (float): Minimum cutout size.
            size_max (float): Maximum cutout size.
            ratio_1 (float): Minimum aspect ratio.
            ratio_2 (float): Maximum aspect ratio.
            value_min (int): Minimum pixel value for cutout.
            value_max (int): Maximum pixel value for cutout.
            pixel_level (bool): Whether to apply cutout at the pixel level.
    
        Returns:
            PIL Image, PIL Image: Transformed image and mask.
        """
        if random.random() < p:
            img = np.array(img)
            mask = np.array(mask)
    
            img_h, img_w, img_c = img.shape
    
            while True:
                size = np.random.uniform(size_min, size_max) * img_h * img_w
                ratio = np.random.uniform(ratio_1, ratio_2)
                erase_w = int(np.sqrt(size / ratio))
                erase_h = int(np.sqrt(size * ratio))
                x = np.random.randint(0, img_w)
                y = np.random.randint(0, img_h)
    
                if x + erase_w <= img_w and y + erase_h <= img_h:
                    break
    
            if pixel_level:
                value = np.random.uniform(value_min, value_max, (erase_h, erase_w, img_c))
            else:
                value = np.random.uniform(value_min, value_max)
    
            img[y:y + erase_h, x:x + erase_w] = value
            mask[y:y + erase_h, x:x + erase_w] = 0
    
            # img = Image.fromarray(img.astype(np.uint8))
            # mask = Image.fromarray(mask.astype(np.uint8))
    
        return img, mask


    @staticmethod
    def mask_shrinking(img, mask, structure_size=8, iterations=1):
        """
        Shrink the size of classified pixels in a binary mask through erosion and resizing.
    
        Parameters:
        - input_array: Binary array where 1 represents the classified pixels.
        - structure_size: Size of the structuring element for erosion.
        - iterations: Number of iterations for erosion.
    
        Returns:
        - Shrunken mask resized to (256, 256).
        """
        # Calculate the number of pixels to be eroded around the borders
        to_be_eroded = int(np.ceil(structure_size / 2))
        
        # Create a structuring element for binary erosion
        structure = np.ones((structure_size, structure_size, 1)) if len(mask.shape) > 2 else np.ones((structure_size, structure_size))
    
        # Perform binary erosion on the input array
        eroded_mask = binary_erosion(mask, structure=structure, iterations=iterations)
        
        # Crop and resize the eroded mask to remove pixels affected by the border effects
        cropped_eroded_mask = eroded_mask[to_be_eroded:-to_be_eroded, to_be_eroded:-to_be_eroded]
        resized_mask = cv2.resize(cropped_eroded_mask.astype(np.uint8), (256, 256))
    
        # Crop and resize the img
        cropped_img = img[to_be_eroded:-to_be_eroded, to_be_eroded:-to_be_eroded]
        resized_img = cv2.resize(cropped_img, (256, 256))
    
        return resized_img, resized_mask

    def strong_augment(self, image, mask):
        """
        Apply strong data augmentation to an image and its corresponding mask.

        Parameters:
        - image (numpy.ndarray): The input image.
        - mask (numpy.ndarray): The segmentation mask.

        Returns:
        - tuple: A tuple containing the augmented image and the augmented segmentation mask.
        """
        # Convert image to uint8
        img = image.astype('uint8')

        # Convert mask to integer values (0 or 1)
        segmap = (mask / 255).astype(np.int32)

        # if random.random() > 0.5:
        #     img, segmap = self.mask_shrinking(img, segmap)
            
        # Create SegmentationMapsOnImage object
        segmap = SegmentationMapsOnImage(segmap, shape=img.shape)

        # Apply augmentation sequence to image and segmentation map
        image_aug, segmap_aug = self.seq(image=img, segmentation_maps=segmap)

        # Draw segmentation map
        segmap_aug = segmap_aug.draw()[0]

        return image_aug, segmap_aug
        # return img, segmap


class LoadPreprocessImages:
    @staticmethod
    def equalize_histogram(image, convert_to_tensors=True):
        """
        Apply histogram equalization to an RGB image.

        Parameters:
        - image (tf.Tensor): Input RGB image.

        Returns:
        - tf.Tensor: Image after histogram equalization.
        """
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

    @staticmethod
    def adap_hist_equalize(img):
        # histogram equalization
        equalized_image = cv2.equalizeHist(img)
        # Adaptive histogram equalization is supposed to be more robust
        # CLAHE = Contrast Limited Adaptive Histogram Equalization
        # Create a CLAHE object
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
        # Apply CLAHE to the image
        adap_equalized_image = clahe.apply(equalized_image)
        return adap_equalized_image

    @staticmethod
    def load_preprocess_images(rgb, equalize, add_noise, std_noise=7, max_samples=None, img_row=256, img_col=256, gt_chan=1, test_split=0.2, shuffle=True):
        if rgb:
            img_chan = 3
        else:
            img_chan = 1

        img_list, gt_list = get_file_paths()

        if max_samples:
            random_indices = get_random_indices(range(len(img_list)), max_samples)
            random_indices.sort()
            img_list = np.array(img_list)[random_indices].tolist()
            gt_list = np.array(gt_list)[random_indices].tolist()

        num_imgs = len(img_list)

        if rgb:
            imgs = np.zeros((num_imgs, img_row, img_col, 3))
        else:
            imgs = np.zeros((num_imgs, img_row, img_col))
        gts = np.zeros((num_imgs, img_row, img_col))

        for i in tqdm(range(num_imgs)):
            tmp_img = plt.imread(img_list[i])
            tmp_gt = plt.imread(gt_list[i])

            img = cv2.resize(tmp_img, (img_col, img_row), interpolation=cv2.INTER_NEAREST)
            gt = cv2.resize(tmp_gt, (img_col, img_row), interpolation=cv2.INTER_NEAREST)

            if rgb:
                if equalize:
                    img = LoadPreprocessImages.equalize_histogram(img, convert_to_tensors=False)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                if equalize:
                    img = LoadPreprocessImages.adap_hist_equalize(img)

            
            if add_noise:
                gaussian = np.round(np.random.normal(0, std_noise, (img.shape)))
                img = img + gaussian
                # img = (img + noise * img.std() * np.random.random(img.shape)).astype(np.uint8)
                
            

            if img.max() > 1:
                img = img / 255.
            if gt.max() > 1:
                gt = gt / 255.
            
            imgs[i] = img # np.stack([img]*3, axis=2)# img
            gts[i] = gt

        indices = np.arange(0, num_imgs, 1)

        imgs_train, imgs_test, \
        imgs_mask_train, imgs_mask_test, \
        trainIdx, testIdx = train_test_split(imgs, gts, indices, test_size=test_split, shuffle=shuffle)

        if not rgb:
            imgs_train = np.expand_dims(imgs_train, axis=3)
            imgs_test = np.expand_dims(imgs_test, axis=3)

        imgs_mask_train = np.expand_dims(imgs_mask_train, axis=3)
        imgs_mask_test = np.expand_dims(imgs_mask_test, axis=3)

        return imgs_train, imgs_mask_train, imgs_test, imgs_mask_test


class orthosSequence(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, input_img_paths, target_img_paths, rgb, add_noise, year, std_noise=7, segformer=False, img_size=(256, 256)):
        """
        This class is designed to help iterate over data stored as Numpy arrays for model training. It supports various
        options such as adding noise to the images, converting them to grayscale, and working with Segformer format.
        """
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.rgb = rgb
        self.add_noise = add_noise
        self.std_noise = std_noise
        self.img_size = img_size
        self.year = year
        self.segformer = segformer

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        
        if self.rgb:
            x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        else:
            if self.segformer:
                x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
            else:
                x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        # y_4 = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        # y_3 = np.zeros((self.batch_size,) + tuple((np.array(self.img_size)/2).astype(int)) + (1,), dtype="float32") 
        # y_2 = np.zeros((self.batch_size,) + tuple((np.array(self.img_size)/4).astype(int)) + (1,), dtype="float32")
        # y_1 = np.zeros((self.batch_size,) + tuple((np.array(self.img_size)/8).astype(int)) + (1,), dtype="float32")
        
        for j in range(len(batch_input_img_paths)):
            img_path = batch_input_img_paths[j]
            mask_path = batch_target_img_paths[j]
            
            img = load_img(img_path, target_size=self.img_size)
            img = np.array(img)
            
            if not self.rgb:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if self.add_noise:
                if self.rgb:
                    if self.year==2006:
                        # if j%4==1:
                        #     # print(1)
                        #     img = cv2.GaussianBlur(img,(3, 3),cv2.BORDER_DEFAULT)
                        # elif j%4==2:
                        #     # print(2)
                        #     img = cv2.GaussianBlur(img,(5, 5),cv2.BORDER_DEFAULT)
                        # elif j%4==3:
                        #     # print(3)
                        #     img = denoise_wavelet(img, channel_axis=-1)
                        # else:
                        #     # print(0)
                        #     img = img.copy()
                        # img = denoise_tv_chambolle(img, channel_axis=-1)
                        # img = denoise_tv_bregman(img, channel_axis=-1)
                        img = cv2.GaussianBlur(img,(5, 5),cv2.BORDER_DEFAULT)
                        # img = rgb_svd_decomposition(img, k=40)
                        # img = denoise_wavelet(img, channel_axis=-1, convert2ycbcr=True,
                                # rescale_sigma=True)
                        
                    elif self.year==2012:
                        img = denoise_wavelet(img, channel_axis=-1, convert2ycbcr=True,
                                rescale_sigma=True)
                    # img = rgb_svd_decomposition(img, k=int((1/5)*self.img_size[0]))
                else:
                    # img = hist_match(img, template)
                    img = random_noise(img, mode='gaussian', var=0.0015)
                    # img = random_noise(img, mode='speckle')
                    # img = denoise_wavelet(img, rescale_sigma=True)
                    # img = gray_svd_decomposition(img, k=int((1/5)*self.img_size[0]))
                # gaussian = np.round(np.random.normal(0, self.std_noise, (img.shape)))
                # img = img + gaussian
                # img = (img + self.noise * img.std() * np.random.random(img.shape)).astype(np.uint8)
            if img.max() > 1:
                img = img/255.
                
            # img = per_image_standardization(img)
            
            x[j] = img if self.rgb else (np.stack([img]*3, axis=2) if self.segformer else np.expand_dims(img, 2))
        
            mask = load_img(mask_path, target_size=self.img_size, color_mode="grayscale")
            mask = np.asarray(mask)
            if mask.max() > 1:
                mask = mask/255.
            mask = np.expand_dims(mask, 2)
            y[j] = mask
        return x, y
            
#             mask_4 = np.expand_dims(mask, 2)
#             mask_3 = mask_4[::2,::2,:]
#             mask_2 = mask_3[::2,::2,:]
#             mask_1 = mask_2[::2,::2,:]
            
#             y_4[j] = mask_4
#             y_3[j] = mask_3
#             y_2[j] = mask_2
#             y_1[j] = mask_1
            
#         return x, [y_1, y_2, y_3, y_4]


class AugmentedOrthosSequence(keras.utils.Sequence):
    def __init__(self, batch_size, input_img_paths, target_img_paths, rgb, add_noise, year, augment='differentiated', img_size=(256, 256), shrinking_mode='always', shrinking_structure_size = 8):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.rgb = rgb
        self.add_noise = add_noise
        self.img_size = img_size
        self.year = year
        self.augment = augment
        self.shrinking_mode = shrinking_mode
        self.shrinking_structure_size = shrinking_structure_size
        self.augmentation = Augmentation()

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        
        if self.rgb:
            x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        else:
            x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        
        for j in range(len(batch_input_img_paths)):
            img_path = batch_input_img_paths[j]
            mask_path = batch_target_img_paths[j]

            year = re.findall(r'[0-9]+', img_path)[0]
            
            img = load_img(img_path, target_size=self.img_size)
            img = np.array(img)
            # print(img.shape)
            
            if not self.rgb:
                if int(year) <= 1993:
                    img = img[:,:,0]
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
            mask = load_img(mask_path, target_size=self.img_size, color_mode="grayscale")
            mask = np.array(mask)

            # print(img.shape)
            if self.shrinking_mode == 'always':
                img, mask = self.augmentation.mask_shrinking(img, mask, structure_size=self.shrinking_structure_size)
            elif self.shrinking_mode == 'random':
                if random.random() > 0.4:
                    img, mask = self.augmentation.mask_shrinking(img, mask, structure_size=self.shrinking_structure_size)
            elif self.shrinking_mode == 'differentiated':
                if year != '2020':
                    img, mask = self.augmentation.mask_shrinking(img, mask, structure_size=self.shrinking_structure_size)

            if mask.max() > 1:
                mask = mask/255.
                
            if self.augment == 'differentiated':
                # basic augmentation on all training images
                img, mask = self.augmentation.resize(img, mask, 256, (0.5, 2.0))
                img, mask = self.augmentation.crop(img, mask, 256)
                img, mask = self.augmentation.hflip(img, mask, p=.5)
                if year != '2020':
                    # strong augmentation
                    img = self.augmentation.colorjitter(img)
                    img = self.augmentation.to_grayscale_converter(img)# .astype(int)
                    img = self.augmentation.blur(img, p=0.5)
                    img, mask = self.augmentation.cutout(img, mask, p=.5)
                
            if img.max() > 1:
                img = img/255.

            
            x[j] = np.expand_dims(img, 2) if not self.rgb else img
            y[j] = np.expand_dims(np.round(mask), 2) # mask
            # print('')
            
        return x, y

