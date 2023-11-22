from utils import get_random_indices, get_file_paths, gray_svd_decomposition, rgb_svd_decomposition# , hist_match

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split


from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2 
from skimage.restoration import denoise_wavelet# , denoise_tv_bregman, denoise_tv_chambolle
from skimage.util import random_noise
# from PIL import Image

import warnings
warnings.simplefilter("ignore")


# from keras import backend as K

# K.set_image_data_format('channels_last')  # Set the data format to 'channels_last' (TensorFlow dimension ordering)



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
                        # img = cv2.GaussianBlur(img,(5, 5),cv2.BORDER_DEFAULT)
                        # img = rgb_svd_decomposition(img, k=40)
                        img = denoise_wavelet(img, channel_axis=-1, convert2ycbcr=True,
                                rescale_sigma=True)
                        
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

