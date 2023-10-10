from utils import get_random_indices, get_training_file_paths, gray_svd_decomposition

import tensorflow as tf 
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Activation, add, multiply, Lambda
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
from keras.initializers import glorot_normal
from keras.layers import AveragePooling2D
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split

from keras import backend as K


import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2 
from skimage import color
# from PIL import Image

import warnings
warnings.simplefilter("ignore")

with open('config.json') as f_in:
    config = json.load(f_in)

K.set_image_data_format('channels_last')  # Set the data format to 'channels_last' (TensorFlow dimension ordering)


class AttentionUnet:
    def __init__(self, input_size):
        self.input_size = input_size
        # Define weight initialization method
        self.kinit = 'glorot_normal'

    def expend_as(self, tensor, rep, name):
        my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep}, name='psi_up'+name)(tensor)
        return my_repeat

    def AttnGatingBlock(self, x, g, inter_shape, name):
        # Get shapes of 'x' and 'g'
        shape_x = K.int_shape(x)  # 32
        shape_g = K.int_shape(g)  # 16

        # Convolution on 'x' to match dimensions
        theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same', name='xl'+name)(x)  # 16
        # Get shape of the convolved 'theta_x'
        shape_theta_x = K.int_shape(theta_x)

        # Convolution on 'g' to match dimensions
        phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
        # Upsample 'g' to match 'theta_x' dimensions
        upsample_g = Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same', name='g_up'+name)(phi_g)  # 16

        # Concatenate upsampled 'g' and 'theta_x'
        concat_xg = add([upsample_g, theta_x])

        # Apply ReLU activation
        act_xg = Activation('relu')(concat_xg)
        # Calculate attention coefficients ('psi')
        psi = Conv2D(1, (1, 1), padding='same', name='psi'+name)(act_xg)
        # Apply sigmoid activation to obtain attention coefficients
        sigmoid_xg = Activation('sigmoid')(psi)
        # Get shape of sigmoid activation
        shape_sigmoid = K.int_shape(sigmoid_xg)
         # Upsample attention coefficients to match original spatial dimensions of 'x'
        upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

        # Expand attention coefficients to match the number of channels in 'x'
        upsample_psi = self.expend_as(upsample_psi, shape_x[3],  name)
        # Multiply attention coefficients and 'x' to obtain attended feature map
        y = multiply([upsample_psi, x], name='q_attn'+name)

        # Convolution on the attended feature map
        result = Conv2D(shape_x[3], (1, 1), padding='same',name='q_attn_conv'+name)(y)
        # Batch normalization on the result
        result_bn = BatchNormalization(name='q_attn_bn'+name)(result)
        return result_bn

    def UnetConv2D(self, input, outdim, is_batchnorm, name):
        x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=self.kinit, padding="same", name=name+'_1')(input)
        if is_batchnorm:
            x = BatchNormalization(name=name + '_1_bn')(x)
        x = Activation('relu',name=name + '_1_act')(x)
        
        x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=self.kinit, padding="same", name=name+'_2')(x)
        if is_batchnorm:
            x = BatchNormalization(name=name + '_2_bn')(x)
        x = Activation('relu', name=name + '_2_act')(x)
        return x

    def UnetGatingSignal(self, input, is_batchnorm, name):
        # Get the shape of the input tensor
        shape = K.int_shape(input)
        # 1x1 Convolution:
        x = Conv2D(shape[3] * 1, (1, 1), strides=(1, 1), padding="same",  kernel_initializer=self.kinit, name=name + '_conv')(input)
        if is_batchnorm:
            # Batch Normalization (optional):
            x = BatchNormalization(name=name + '_bn')(x)
        # ReLU Activation:
        x = Activation('relu', name = name + '_act')(x)
        return x

    def build_model(self):
        '''
        This function defines the complete Attention U-Net architecture.
        It includes encoder and decoder parts with attention mechanisms.
        '''
        # Input layer
        inputs = Input(shape=self.input_size)

        # Encoder part
        conv1 = self.UnetConv2D(inputs, 32, is_batchnorm=True, name='conv1')
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.UnetConv2D(pool1, 32, is_batchnorm=True, name='conv2')
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.UnetConv2D(pool2, 64, is_batchnorm=True, name='conv3')
        #conv3 = Dropout(0.2,name='drop_conv3')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.UnetConv2D(pool3, 64, is_batchnorm=True, name='conv4')
        #conv4 = Dropout(0.2, name='drop_conv4')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        # Central part
        center = self.UnetConv2D(pool4, 128, is_batchnorm=True, name='center')

        # Gating signals
        g1 = self.UnetGatingSignal(center, is_batchnorm=True, name='g1')
        attn1 = self.AttnGatingBlock(conv4, g1, 128, '_1')
        up1 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=self.kinit)(center), attn1], name='up1')

        g2 = self.UnetGatingSignal(up1, is_batchnorm=True, name='g2')
        attn2 = self.AttnGatingBlock(conv3, g2, 64, '_2')
        up2 = concatenate([Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=self.kinit)(up1), attn2], name='up2')

        g3 = self.UnetGatingSignal(up1, is_batchnorm=True, name='g3')
        attn3 = self.AttnGatingBlock(conv2, g3, 32, '_3')
        up3 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=self.kinit)(up2), attn3], name='up3')

        # Final upsampling and output
        up4 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=self.kinit)(up3), conv1], name='up4')
        out = Conv2D(1, (1, 1), activation='sigmoid',  kernel_initializer=self.kinit, name='final')(up4)

        # Define the model with inputs and outputs
        model = Model(inputs=[inputs], outputs=[out])

        return model



class Losses:
    @staticmethod
    def DiceBCELoss(y_true, y_pred, smooth=1e-6): 
        """
        Calculate the Dice loss combined with Binary Cross-Entropy (BCE) loss.

        This function computes the Dice loss and BCE loss for binary segmentation tasks and combines them.

        Parameters:
        - y_true (tensor): True binary labels (ground truth).
        - y_pred (tensor): Predicted binary labels.
        - smooth (float): Smoothing factor to prevent division by zero.

        Returns:
        - tensor: Combined loss value.
        """
        # Cast to float32 datatype
        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')
        # Flatten label and prediction tensors
        y_pred = K.flatten(y_pred)
        y_true = K.flatten(y_true)

        # BCE
        BCE = keras.losses.BinaryCrossentropy()(y_true, y_pred)

        # Dice loss
        intersection = K.sum(y_true * y_pred)
        dice_loss = 1 - (2 * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

        # Combining
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

    @staticmethod
    def tversky(y_true, y_pred, alpha=0.7, smooth=1e-6):
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

    @staticmethod
    def focal_tversky(y_true, y_pred, gamma=1.33, smooth=1e-6):
        pt_1 = Losses.tversky(y_true, y_pred, smooth=smooth)
        return K.pow((1 - pt_1), gamma)

    @staticmethod
    def Combo_loss(y_true, y_pred, ce_w=0.5, ce_d_w=0.5, smooth=1):
        e = K.epsilon()
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        d = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        y_pred_f = K.clip(y_pred_f, e, 1.0 - e)
        out = - (ce_w * y_true_f * K.log(y_pred_f)) + ((1 - ce_w) * (1.0 - y_true_f) * K.log(1.0 - y_pred_f))
        weighted_ce = K.mean(out, axis=-1)
        combo = (ce_d_w * weighted_ce) - ((1 - ce_d_w) * d)
        return combo


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

        img_list, gt_list = get_training_file_paths()

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

    def __init__(self, batch_size, input_img_paths, target_img_paths, rgb, add_noise, std_noise=7, segformer=False, img_size=(256, 256), smooth = 0):
        
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.smooth = smooth
        self.rgb = rgb
        self.add_noise = add_noise
        self.std_noise = std_noise
        self.img_size = img_size
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
        for j in range(len(batch_input_img_paths)):
            img_path = batch_input_img_paths[j]
            mask_path = batch_target_img_paths[j]
            img = load_img(img_path, target_size=self.img_size)
            img = np.asarray(img)
            if not self.rgb:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if self.add_noise:
                img = gray_svd_decomposition(img, k=int((1/5)*self.img_size[0]))
                # gaussian = np.round(np.random.normal(0, self.std_noise, (img.shape)))
                # img = img + gaussian
                # img = (img + self.noise * img.std() * np.random.random(img.shape)).astype(np.uint8)
            if img.max() > 1:
                img = img/255.
            x[j] = img if self.rgb else (np.stack([img]*3, axis=2) if self.segformer else np.expand_dims(img, 2))#gaussian_filter(img, sigma=(self.smooth,self.smooth,0))
        
            img = load_img(mask_path, target_size=self.img_size, color_mode="grayscale")
            img = np.asarray(img)
            if img.max() > 1:
                img = img/255.
            y[j] = np.expand_dims(img, 2)
            
        return x, y
