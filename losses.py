# Tensorflow
from tensorflow import keras
from tensorflow.keras import backend as K



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