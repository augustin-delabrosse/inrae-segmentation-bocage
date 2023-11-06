import tensorflow as tf 
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Activation, add, multiply, Lambda
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from keras.initializers import glorot_normal

from keras import backend as K

import warnings
warnings.simplefilter("ignore")

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

