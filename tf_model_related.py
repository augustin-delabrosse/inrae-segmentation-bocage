import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 
from tensorflow import keras
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, AveragePooling2D
from keras.layers import Activation, add, multiply, Lambda
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from keras.initializers import glorot_normal
from keras.layers import Layer

from keras import backend as K

import warnings
warnings.simplefilter("ignore")

K.set_image_data_format('channels_last')  # Set the data format to 'channels_last' (TensorFlow dimension ordering)



# inspired from https://datascience.stackexchange.com/questions/58884/how-to-create-custom-activation-functions-in-keras-tensorflow#comment119171_66358
def tv_sigmoid(x):
    return K.sigmoid(x)
    
class TVSigmoid(Layer):
    def __init__(self, mu=1.0, trainable=False, **kwargs):
        super(TVSigmoid, self).__init__(**kwargs)
        self.supports_masking = True
        self.mu = mu
        self.trainable = trainable

    def build(self, input_shape):
        self.mu_factor = K.variable(self.mu,
                                      dtype=K.floatx(),
                                      name='mu_factor')
        if self.trainable:
            self._trainable_weights.append(self.mu_factor)

        super(TVSigmoid, self).build(input_shape)

    def call(self, inputs):
        return tv_sigmoid(inputs)

    def get_config(self):
        config = {'mu': self.get_weights()[0] if self.trainable else self.mu,
                  'trainable': self.trainable}
        base_config = super(TVSigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
    

# https://github.com/cpuimage/MaxDropout/tree/master
class MaxDropout(tf.keras.layers.Layer):
    """MaxDropout: Deep Neural Network Regularization Based on Maximum Output Values
    (https://arxiv.org/abs/2007.13723)
    """

    def __init__(self, rate=0.3, trainable=True, name=None, **kwargs):
        super(MaxDropout, self).__init__(name=name, trainable=trainable, **kwargs)
        if rate < 0 or rate > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(rate))
        self.rate = 1. - rate

    def call(self, inputs, training=None):
        if training:
            min_in = tf.math.reduce_min(inputs)
            max_in = tf.math.reduce_max(inputs)
            up = inputs - min_in
            divisor = max_in - min_in
            inputs_out = tf.math.divide_no_nan(up, divisor)
            return tf.where(inputs_out > self.rate, tf.zeros_like(inputs), inputs_out)
        else:
            return inputs
        
# https://github.com/nabsabraham/focal-tversky-unet/tree/master   
class AttentionUnet:
    def __init__(self, input_size=(256, 256, 3)):
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
        upsample_psi = self.expend_as(upsample_psi, shape_x[3],  name) # Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': shape_x[3]}, name='psi_up'+name)(upsample_psi)# 
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


    def build_attention_unet(self):
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
        # out = Conv2D(2, (1, 1), activation='softmax',  kernel_initializer=self.kinit, name='final')(up4)

        # Define the model with inputs and outputs
        model = Model(inputs=[inputs], outputs=[out])

        return model
    
    def build_improved_attention_unet(self):
        img_input = Input(shape=self.input_size, name='input_scale1')
        scale_img_2 = AveragePooling2D(pool_size=(2, 2), name='input_scale2')(img_input)
        scale_img_3 = AveragePooling2D(pool_size=(2, 2), name='input_scale3')(scale_img_2)
        scale_img_4 = AveragePooling2D(pool_size=(2, 2), name='input_scale4')(scale_img_3)

        conv1 = self.UnetConv2D(img_input, 32, is_batchnorm=True, name='conv1')
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        input2 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv_scale2')(scale_img_2)
        input2 = concatenate([scale_img_2, input2, pool1], axis=3)
        conv2 = self.UnetConv2D(input2, 64, is_batchnorm=True, name='conv2')
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        input3 = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv_scale3')(scale_img_3)
        input3 = concatenate([scale_img_3, input3, pool2], axis=3)
        conv3 = self.UnetConv2D(input3, 128, is_batchnorm=True, name='conv3')
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        input4 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv_scale4')(scale_img_4)
        input4 = concatenate([scale_img_4, input4, pool3], axis=3)
        conv4 = self.UnetConv2D(input4, 64, is_batchnorm=True, name='conv4')
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        center = self.UnetConv2D(pool4, 512, is_batchnorm=True, name='center')

        g1 = self.UnetGatingSignal(center, is_batchnorm=True, name='g1')
        attn1 = self.AttnGatingBlock(conv4, g1, 128, '_1')
        up1 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=self.kinit)(center), attn1], name='up1')

        g2 = self.UnetGatingSignal(up1, is_batchnorm=True, name='g2')
        attn2 = self.AttnGatingBlock(conv3, g2, 64, '_2')
        up2 = concatenate([Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=self.kinit)(up1), attn2], name='up2')

        g3 = self.UnetGatingSignal(up1, is_batchnorm=True, name='g3')
        attn3 = self.AttnGatingBlock(conv2, g3, 32, '_3')
        up3 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=self.kinit)(up2), attn3], name='up3')

        up4 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=self.kinit)(up3), conv1], name='up4')

        conv6 = self.UnetConv2D(up1, 256, is_batchnorm=True, name='conv6')
        conv7 = self.UnetConv2D(up2, 128, is_batchnorm=True, name='conv7')
        conv8 = self.UnetConv2D(up3, 64, is_batchnorm=True, name='conv8')
        conv9 = self.UnetConv2D(up4, 32, is_batchnorm=True, name='conv9')

        out6 = Conv2D(1, (1, 1), activation='sigmoid', name='pred1')(conv6)
        out7 = Conv2D(1, (1, 1), activation='sigmoid', name='pred2')(conv7)
        out8 = Conv2D(1, (1, 1), activation='sigmoid', name='pred3')(conv8)
        out9 = Conv2D(1, (1, 1), activation='sigmoid', name='final')(conv9)

        model = Model(inputs=[img_input], outputs=[out6, out7, out8, out9])

        return model

    def build_regularized_attention_unet(self, drop_rate=0.3):
        '''
        This function defines the complete Attention U-Net architecture.
        It includes encoder and decoder parts with attention mechanisms.
        '''
        # Input layer
        inputs = Input(shape=self.input_size)

        # Encoder part
        conv1 = self.UnetConv2D(inputs, 32, is_batchnorm=True, name='conv1')
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        drop1 = MaxDropout(rate=drop_rate)(pool1)

        conv2 = self.UnetConv2D(drop1, 32, is_batchnorm=True, name='conv2')
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        drop2 = MaxDropout(rate=drop_rate)(pool2)
        
        conv3 = self.UnetConv2D(drop2, 64, is_batchnorm=True, name='conv3')
        #conv3 = Dropout(0.2,name='drop_conv3')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        drop3 = MaxDropout(rate=drop_rate)(pool3)

        conv4 = self.UnetConv2D(drop3, 64, is_batchnorm=True, name='conv4')
        #conv4 = Dropout(0.2, name='drop_conv4')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        drop4 = MaxDropout(rate=drop_rate)(pool4)
        
        # Central part
        center = self.UnetConv2D(drop4, 128, is_batchnorm=True, name='center')

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
        out = Conv2D(1, (1, 1), kernel_initializer=self.kinit, name='final')(up4)
        # out = Conv2D(1, (1, 1), activation='sigmoid',  kernel_initializer=self.kinit, name='final')(up4)
        out = TVSigmoid(mu=1.0, trainable=True)(out)

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
        """
        Calculate the Tversky loss for binary segmentation tasks.

        This function computes the Tversky loss, a generalization of the Dice loss, with user-defined parameters.

        Parameters:
        - y_true (tensor): True binary labels (ground truth).
        - y_pred (tensor): Predicted binary labels.
        - alpha (float): Tversky index parameter.
        - smooth (float): Smoothing factor to prevent division by zero.

        Returns:
        - tensor: Tversky loss value.
        """
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

    @staticmethod
    def focal_tversky(y_true, y_pred, gamma=1.33, smooth=1e-6):
        """
        Calculate the Focal Tversky loss for binary segmentation tasks.

        This function computes the Focal Tversky loss, which is a variant of the Tversky loss with a focal parameter.

        Parameters:
        - y_true (tensor): True binary labels (ground truth).
        - y_pred (tensor): Predicted binary labels.
        - gamma (float): Focal parameter.
        - smooth (float): Smoothing factor to prevent division by zero.

        Returns:
        - tensor: Focal Tversky loss value.
        """
        pt_1 = Losses.tversky(y_true, y_pred, smooth=smooth)
        return K.pow((1 - pt_1), gamma)

    @staticmethod
    def focal_tversky_ignore_index(y_true, y_pred, gamma=1.33, smooth=1e-6, ignore_index=255):
        """
        Calculate the Focal Tversky loss for binary segmentation tasks.

        This function computes the Focal Tversky loss, which is a variant of the Tversky loss with a focal parameter.

        Parameters:
        - y_true (tensor): True binary labels (ground truth).
        - y_pred (tensor): Predicted binary labels.
        - gamma (float): Focal parameter.
        - smooth (float): Smoothing factor to prevent division by zero.
        - ignore_index (int): Pixel value to be ignored in the loss calculation.

        Returns:
        - tensor: Focal Tversky loss value.
        """
        if ignore_index is not None:
            mask = K.cast(K.not_equal(y_true, ignore_index), K.floatx())
            # print(mask)
            y_true = y_true * mask
            # print(y_true)
            y_pred = y_pred * mask
            # print(y_pred)

        pt_1 = Losses.tversky(y_true, y_pred, smooth=smooth)
        return K.pow((1 - pt_1), gamma)


    @staticmethod
    def Combo_loss(y_true, y_pred, ce_w=0.5, ce_d_w=0.5, smooth=1):
        """
        Calculate the Combo loss for binary segmentation tasks.

        This function computes the Combo loss, which combines Binary Cross-Entropy (BCE) and Dice losses with user-defined weights.

        Parameters:
        - y_true (tensor): True binary labels (ground truth).
        - y_pred (tensor): Predicted binary labels.
        - ce_w (float): Weight for BCE loss.
        - ce_d_w (float): Weight for Dice loss.
        - smooth (float): Smoothing factor to prevent division by zero.

        Returns:
        - tensor: Combo loss value.
        """
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



class RegularizedAttentionUnet(AttentionUnet):
    def __init__(self, input_size=(256, 256, 3), dropout_rate=0.3, l2_reg=1e-4):
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
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
        upsample_psi = self.expend_as(upsample_psi, shape_x[3],  name) # Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': shape_x[3]}, name='psi_up'+name)(upsample_psi)# 
        # Multiply attention coefficients and 'x' to obtain attended feature map
        y = multiply([upsample_psi, x], name='q_attn'+name)

        # Convolution on the attended feature map
        result = Conv2D(shape_x[3], (1, 1), padding='same',name='q_attn_conv'+name)(y)
        # Batch normalization on the result
        result_bn = BatchNormalization(name='q_attn_bn'+name)(result)
        return result_bn

    def UnetConv2D(self, input, outdim, is_batchnorm, name):
        x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=self.kinit, padding="same", 
                   kernel_regularizer=l2(self.l2_reg), name=name+'_1')(input)
        if is_batchnorm:
            x = BatchNormalization(name=name + '_1_bn')(x)
        x = Activation('relu', name=name + '_1_act')(x)
        x = Dropout(self.dropout_rate, name='drop_' + name + '_1')(x)

        x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=self.kinit, padding="same",
                   kernel_regularizer=l2(self.l2_reg), name=name+'_2')(x)
        if is_batchnorm:
            x = BatchNormalization(name=name + '_2_bn')(x)
        x = Activation('relu', name=name + '_2_act')(x)
        x = Dropout(self.dropout_rate, name='drop_' + name + '_2')(x)
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

#     def build_attention_unet(self):
#         # Input layer
#         inputs = Input(shape=self.input_size)

#         # Encoder part
#         conv1 = self.UnetConv2D(inputs, 32, is_batchnorm=True, name='conv1')
#         pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

#         conv2 = self.UnetConv2D(pool1, 32, is_batchnorm=True, name='conv2')
#         pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

#         conv3 = self.UnetConv2D(pool2, 64, is_batchnorm=True, name='conv3')
#         pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

#         conv4 = self.UnetConv2D(pool3, 64, is_batchnorm=True, name='conv4')
#         pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

#         # Central part
#         center = self.UnetConv2D(pool4, 128, is_batchnorm=True, name='center')

#         # Gating signals
#         g1 = self.UnetGatingSignal(center, is_batchnorm=True, name='g1')
#         attn1 = self.AttnGatingBlock(conv4, g1, 128, '_1')
#         up1 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=self.kinit)(center), attn1], name='up1')

#         g2 = self.UnetGatingSignal(up1, is_batchnorm=True, name='g2')
#         attn2 = self.AttnGatingBlock(conv3, g2, 64, '_2')
#         up2 = concatenate([Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=self.kinit)(up1), attn2], name='up2')

#         g3 = self.UnetGatingSignal(up1, is_batchnorm=True, name='g3')
#         attn3 = self.AttnGatingBlock(conv2, g3, 32, '_3')
#         up3 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=self.kinit)(up2), attn3], name='up3')

#         # Final upsampling and output
#         up4 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=self.kinit)(up3), conv1], name='up4')
#         out = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer=self.kinit, name='final')(up4)

#         # Define the model with inputs and outputs
#         model = Model(inputs=[inputs], outputs=[out])

#         return model
    
    
    def build_attention_unet(self):
        # Input layer
        inputs = Input(shape=self.input_size)

        # Encoder part
        conv1 = self.UnetConv2D(inputs, 32, is_batchnorm=True, name='conv1')
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.UnetConv2D(pool1, 32, is_batchnorm=True, name='conv2')
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.UnetConv2D(pool2, 64, is_batchnorm=True, name='conv3')
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # conv4 = self.UnetConv2D(pool3, 64, is_batchnorm=True, name='conv4')
        # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        # Central part
        center = self.UnetConv2D(pool3, 128, is_batchnorm=True, name='center')

        # Gating signals
        # g1 = self.UnetGatingSignal(center, is_batchnorm=True, name='g1')
        # attn1 = self.AttnGatingBlock(conv4, g1, 128, '_1')
        # up1 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=self.kinit)(center), attn1], name='up1')

        g2 = self.UnetGatingSignal(center, is_batchnorm=True, name='g2')
        attn2 = self.AttnGatingBlock(conv3, g2, 64, '_2')
        up2 = concatenate([Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=self.kinit)(center), attn2], name='up2')

        g3 = self.UnetGatingSignal(up2, is_batchnorm=True, name='g3')
        attn3 = self.AttnGatingBlock(conv2, g3, 32, '_3')
        up3 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=self.kinit)(up2), attn3], name='up3')

        # Final upsampling and output
        up4 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=self.kinit)(up3), conv1], name='up4')
        out = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer=self.kinit, name='final')(up4)

        # Define the model with inputs and outputs
        model = Model(inputs=[inputs], outputs=[out])

        return model

class PerformancePlotCallback(keras.callbacks.Callback):
    def __init__(self, val_gen, batch):
        batch = val_gen.__getitem__(batch)
        self.imgs = batch[0]
        self.trues = batch[1]
        
    def on_epoch_end(self, epoch, logs={}):
        # clear_output(wait=True)
        y_preds = self.model.predict(self.imgs)
        plt.figure(figsize=(12, 12))

        title = ["Input Image", "True Mask", "Predicted Mask"]
        display_list = [self.imgs[0], self.trues[0], np.round(y_preds[0])]
        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i + 1)
            plt.title(title[i])
            plt.imshow(keras.utils.array_to_img(display_list[i]))
            plt.axis("off")
        plt.show()
        print("\nSample Prediction after epoch {}\n".format(epoch + 1))

class SaveCheckpointAtEpoch(keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, fname, save_epochs):
        super(SaveCheckpointAtEpoch, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.fname = fname
        self.save_epochs = save_epochs

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) in self.save_epochs:
            model_checkpoint_path = os.path.join(self.checkpoint_dir, 
                                                 self.fname.replace('epoch1', str(epoch+1)))
            self.model.save(model_checkpoint_path)
            print(f"""
            Model checkpoint saved at epoch {epoch+1}.
            """)