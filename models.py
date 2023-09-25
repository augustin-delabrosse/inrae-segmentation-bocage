import tensorflow as tf 
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Activation, add, multiply, Lambda
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
from keras.initializers import glorot_normal
from keras.callbacks import ModelCheckpoint

from keras import backend as K

K.set_image_data_format('channels_last')  # Set the data format to 'channels_last' (TensorFlow dimension ordering)
# Define weight initialization method
kinit = 'glorot_normal'

# Utility function to repeat tensor elements
@keras.saving.register_keras_serializable(package="MyLayers")
def expend_as(tensor, rep, name):
	my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep},  name='psi_up'+name)(tensor)
	return my_repeat

# Function to create an Attention Gating Block
@keras.saving.register_keras_serializable(package="MyLayers")
def AttnGatingBlock(x, g, inter_shape, name):
    ''' 
    This function takes two inputs, 'x' and 'g', and calculates attention coefficients.
    It performs the following steps:
    1. Convolutions are applied to 'x' and 'g' to match their channel dimensions.
    2. 'g' is upsampled to have the same spatial dimensions as 'x'.
    3. The upsampled 'g' and 'x' are concatenated.
    4. Activation functions (ReLU and sigmoid) are applied to obtain attention coefficients.
    5. The attention coefficients are upsampled to match the original spatial dimensions of 'x'.
    6. Element-wise multiplication is performed between the upsampled attention coefficients and 'x' to obtain the attended feature map.
    '''
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
    upsample_psi = expend_as(upsample_psi, shape_x[3],  name)
    # Multiply attention coefficients and 'x' to obtain attended feature map
    y = multiply([upsample_psi, x], name='q_attn'+name)

    # Convolution on the attended feature map
    result = Conv2D(shape_x[3], (1, 1), padding='same',name='q_attn_conv'+name)(y)
    # Batch normalization on the result
    result_bn = BatchNormalization(name='q_attn_bn'+name)(result)
    return result_bn

# Function to define a U-Net convolution block
@keras.saving.register_keras_serializable(package="MyLayers")
def UnetConv2D(input, outdim, is_batchnorm, name):
    '''
    This function defines a U-Net convolution block.
    It performs two convolution operations, each followed by batch normalization and ReLU activation.
    '''
	
    x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name+'_1')(input)
    if is_batchnorm:
        x =- BatchNormalization(name=name + '_1_bn')(x)
    x = Activation('relu',name=name + '_1_act')(x)
    
    x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name+'_2')(x)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_act')(x)
    return x
	
# Function to create a U-Net gating signal
@keras.saving.register_keras_serializable(package="MyLayers")
def UnetGatingSignal(input, is_batchnorm, name):
    '''
    This function creates a U-Net gating signal.
    It performs a 1x1 convolution, batch normalization (optional), and ReLU activation.
    '''
    # Get the shape of the input tensor
    shape = K.int_shape(input)
    # 1x1 Convolution:
    x = Conv2D(shape[3] * 1, (1, 1), strides=(1, 1), padding="same",  kernel_initializer=kinit, name=name + '_conv')(input)
    if is_batchnorm:
        # Batch Normalization (optional):
        x = BatchNormalization(name=name + '_bn')(x)
    # ReLU Activation:
    x = Activation('relu', name = name + '_act')(x)
    return x

# Function to define the complete Attention U-Net model 
@keras.saving.register_keras_serializable(package="MyModel")
def attn_unet(input_size): 
    '''
    This function defines the complete Attention U-Net architecture.
    It includes encoder and decoder parts with attention mechanisms.
    '''
    # Input layer
    inputs = Input(shape=input_size)
    
    # Encoder part
    conv1 = UnetConv2D(inputs, 32, is_batchnorm=True, name='conv1')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = UnetConv2D(pool1, 32, is_batchnorm=True, name='conv2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = UnetConv2D(pool2, 64, is_batchnorm=True, name='conv3')
    #conv3 = Dropout(0.2,name='drop_conv3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = UnetConv2D(pool3, 64, is_batchnorm=True, name='conv4')
    #conv4 = Dropout(0.2, name='drop_conv4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Central part
    center = UnetConv2D(pool4, 128, is_batchnorm=True, name='center')

    # Gating signals
    g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')
    attn1 = AttnGatingBlock(conv4, g1, 128, '_1')
    # Upsampling and attention blocks
    up1 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(center), attn1], name='up1')

    g2 = UnetGatingSignal(up1, is_batchnorm=True, name='g2')
    attn2 = AttnGatingBlock(conv3, g2, 64, '_2')
    up2 = concatenate([Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up1), attn2], name='up2')

    g3 = UnetGatingSignal(up1, is_batchnorm=True, name='g3')
    attn3 = AttnGatingBlock(conv2, g3, 32, '_3')
    up3 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up2), attn3], name='up3')

    # Final upsampling and output
    up4 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up3), conv1], name='up4')
    out = Conv2D(1, (1, 1), activation='sigmoid',  kernel_initializer=kinit, name='final')(up4)

    # Define the model with inputs and outputs
    model = Model(inputs=[inputs], outputs=[out])
    
    return model


@keras.saving.register_keras_serializable(package="MyModel")
def attn_reg(input_size):
    
    img_input = Input(shape=input_size, name='input_scale1')
    scale_img_2 = AveragePooling2D(pool_size=(2, 2), name='input_scale2')(img_input)
    scale_img_3 = AveragePooling2D(pool_size=(2, 2), name='input_scale3')(scale_img_2)
    scale_img_4 = AveragePooling2D(pool_size=(2, 2), name='input_scale4')(scale_img_3)

    conv1 = UnetConv2D(img_input, 32, is_batchnorm=True, name='conv1')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    input2 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv_scale2')(scale_img_2)
    input2 = concatenate([scale_img_2, input2, pool1], axis=3)
    conv2 = UnetConv2D(input2, 64, is_batchnorm=True, name='conv2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    input3 = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv_scale3')(scale_img_3)
    input3 = concatenate([scale_img_3, input3, pool2], axis=3)
    conv3 = UnetConv2D(input3, 128, is_batchnorm=True, name='conv3')
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    input4 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv_scale4')(scale_img_4)
    input4 = concatenate([scale_img_4, input4, pool3], axis=3)
    conv4 = UnetConv2D(input4, 64, is_batchnorm=True, name='conv4')
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        
    center = UnetConv2D(pool4, 512, is_batchnorm=True, name='center')
    
    g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')
    attn1 = AttnGatingBlock(conv4, g1, 128, '_1')
    up1 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(center), attn1], name='up1')

    g2 = UnetGatingSignal(up1, is_batchnorm=True, name='g2')
    attn2 = AttnGatingBlock(conv3, g2, 64, '_2')
    up2 = concatenate([Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up1), attn2], name='up2')

    g3 = UnetGatingSignal(up1, is_batchnorm=True, name='g3')
    attn3 = AttnGatingBlock(conv2, g3, 32, '_3')
    up3 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up2), attn3], name='up3')

    up4 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up3), conv1], name='up4')
    
    conv6 = UnetConv2D(up1, 256, is_batchnorm=True, name='conv6')
    conv7 = UnetConv2D(up2, 128, is_batchnorm=True, name='conv7')
    conv8 = UnetConv2D(up3, 64, is_batchnorm=True, name='conv8')
    conv9 = UnetConv2D(up4, 32, is_batchnorm=True, name='conv9')

    out6 = Conv2D(1, (1, 1), activation='sigmoid', name='pred1')(conv6)
    out7 = Conv2D(1, (1, 1), activation='sigmoid', name='pred2')(conv7)
    out8 = Conv2D(1, (1, 1), activation='sigmoid', name='pred3')(conv8)
    out9 = Conv2D(1, (1, 1), activation='sigmoid', name='final')(conv9)

    model = Model(inputs=[img_input], outputs=[out6, out7, out8, out9])

    return model


def load_custom_model(model_path, custom_objects_list):
    custom_objects_dict = {str(o).split(' ')[1]: o for o in custom_objects_list}
    loaded_model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects_dict
    )
    return loaded_model
    