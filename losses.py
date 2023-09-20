import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import scipy
import numpy as np



# inspired from https://github.com/Mr-TalhaIlyas/Loss-Functions-Package-Tensorflow-Keras-PyTorch
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
    # cast to float32 datatype
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # print(y_true)
    
    #flatten label and prediction tensors
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)

    # BCE
    BCE =  keras.losses.BinaryCrossentropy()(y_true, y_pred)

    # Dice loss
    intersection = intersection = K.sum(y_true * y_pred)
    dice_loss = 1 - (2*intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

    # Combining
    Dice_BCE = BCE + dice_loss
    
    return Dice_BCE


def tversky(y_true, y_pred, alpha=0.7, smooth=1e-6):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def focal_tversky(y_true,y_pred, gamma=1.33, smooth=1e-6):
    pt_1 = tversky(y_true, y_pred, smooth=smooth)
    return K.pow((1-pt_1), gamma)


# def identify_axis(shape):
#     # Three dimensional
#     if len(shape) == 5 : return [1,2,3]
#     # Two dimensional
#     elif len(shape) == 4 : return [1,2]
#     # Exception - Unknown
#     else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')

# def dice_coefficient(delta = 0.5, smooth = 0.000001):
#     """The Dice similarity coefficient, also known as the Sørensen–Dice index or simply Dice coefficient, is a statistical tool which measures the similarity between two sets of data.
#     Parameters
#     ----------
#     delta : float, optional
#         controls weight given to false positive and false negatives, by default 0.5
#     smooth : float, optional
#         smoothing constant to prevent division by zero errors, by default 0.000001
#     """
#     def loss_function(y_true, y_pred):
#         axis = identify_axis(y_true.shape)#y_true.get_shape())
#         # Calculate true positives (tp), false negatives (fn) and false positives (fp)   
#         tp = K.sum(y_true * y_pred, axis=axis)
#         fn = K.sum(y_true * (1-y_pred), axis=axis)
#         fp = K.sum((1-y_true) * y_pred, axis=axis)
#         dice_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
#         # Average class scores
#         dice = K.mean(dice_class)

#         return dice

#     return loss_function

# def combo_loss(alpha=0.5,beta=0.5):
#     """Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation
#     Link: https://arxiv.org/abs/1805.02798
#     Parameters
#     ----------
#     alpha : float, optional
#         controls weighting of dice and cross-entropy loss., by default 0.5
#     beta : float, optional
#         beta > 0.5 penalises false negatives more than false positives., by default 0.5
#     """
#     def loss_function(y_true,y_pred):
#         dice = dice_coefficient()(y_true, y_pred)
#         axis = identify_axis(y_true.shape)#y_true.get_shape())
#         # Clip values to prevent division by zero error
#         epsilon = K.epsilon()
#         y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
#         cross_entropy = -y_true * K.log(y_pred)

#         if beta is not None:
#             beta_weight = tf.constant([beta, 1 - beta], dtype=tf.float32)
#             cross_entropy = beta_weight * cross_entropy
#         # sum over classes
#         cross_entropy = K.mean(K.sum(cross_entropy, axis=[-1]))
#         if alpha is not None:
#             combo_loss = (alpha * cross_entropy) - ((1 - alpha) * dice)
#         else:
#             combo_loss = cross_entropy - dice
#         return combo_loss

#     return loss_function



# def Combo_loss(y_true, y_pred, smooth=1, ALPHA=0.8, CE_RATIO=0.5):
#     e = K.epsilon()
#     # if y_pred.shape[-1] <= 1:
#     # ALPHA = 0.8    # < 0.5 penalises FP more, > 0.5 penalises FN more
#     # CE_RATIO = 0.5 # weighted contribution of modified CE loss compared to Dice loss
#     y_pred = tf.keras.activations.sigmoid(tf.cast(y_pred, 'float32'))  # Cast to float32 and apply Sigmoid
#     # elif y_pred.shape[-1] >= 2:
#     #     ALPHA = 0.3    # < 0.5 penalises FP more, > 0.5 penalises FN more
#     #     CE_RATIO = 0.7 # weighted contribution of modified CE loss compared to Dice loss
#     #     y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
#     #     y_true = K.squeeze(y_true, 3)
#     #     y_true = tf.cast(y_true, "int32")
#     #     y_true = tf.one_hot(y_true, num_class, axis=-1)

#     # Cast to float32 datatype
#     y_true = K.cast(y_true, 'float32')
#     y_pred = K.cast(y_pred, 'float32')

#     targets = K.flatten(y_true)
#     inputs = K.flatten(y_pred)

#     intersection = K.sum(targets * inputs)
#     dice = (2. * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
#     inputs = K.clip(inputs, e, 1.0 - e)
#     out = - (ALPHA * ((targets * K.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * K.log(1.0 - inputs))))
#     weighted_ce = K.mean(out, axis=-1)
#     combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)

#     return combo




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



# Inspired from https://www.sciencedirect.com/science/article/pii/S0303243422000162
# Does not work
# def euclidean_distance_transform(mask):
#     # Calculate the Euclidean Distance Transform for the mask
    
#     edt_mask = scipy.ndimage.distance_transform_edt(mask)
#     edt_mask = tf.cast(edt_mask, tf.float32)

#     return edt_mask

# def loss(y_true, y_pred):
#     # Calculate the Euclidean Distance Transform for y_true
#     d_x = tf.numpy_function(euclidean_distance_transform, [y_true], tf.float32)
    
#     d_x = tf.cast(d_x, tf.float32)
#     sigma = tf.math.reduce_std(y_true)

#     # print(d_x, sigma)
#     # Calculate delta(x) for each pixel
#     delta_x = 1 - tf.exp(-tf.square(d_x) / (2 * sigma**2))

#     # return delta_x
#     # Calculate phi(c) for each class c
#     class_counts = tf.reduce_sum(y_true)  # Count of pixels per class
#     total_pixels = tf.reduce_sum(class_counts)  # Total number of pixels
#     phi_c = total_pixels / (2 * class_counts)

#     # return phi_c
#     # Calculate class weights w(x) for each pixel
#     w_x = phi_c * delta_x

#     # Define the primary loss function, e.g., binary cross-entropy or dice loss
#     primary_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
#     primary_loss = tf.reduce_sum(primary_loss, axis=-1)

#     # return primary_loss
#     # Apply the weighted loss
#     weighted_loss = tf.reduce_mean(w_x * primary_loss)

#     return weighted_loss