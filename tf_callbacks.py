import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model

import warnings
warnings.simplefilter("ignore")

def custom_LearningRate_schedular(epoch, learning_rate=1e-3, max_epoch=None, power=0.9):
    """
    Custom learning rate scheduler function.

    This function calculates the learning rate based on the current epoch using a power decay formula.

    Args:
        epoch (int): Current epoch number.
        learning_rate (float): Initial learning rate.
        max_epoch (int): Total number of epochs.
        power (float): Power parameter for the decay formula.

    Returns:
        float: Updated learning rate for the current epoch.

    Usage:
        learning_rate = custom_LearningRate_schedular(epoch, learning_rate, max_epoch, power)
    """
    if max_epoch is None:
        raise ValueError("max_epoch must be provided.")

    new_lr = learning_rate * tf.math.pow((1 - epoch/max_epoch), power)
    print(f"learning rate : {new_lr}")
    return new_lr


class PerformancePlotCallback(keras.callbacks.Callback):
    """
    Custom Keras callback to plot model performance during training.

    This callback plots input images, true masks, and predicted masks after each epoch.

    Args:
        val_gen: Validation data generator.
        batch (int): Batch index to visualize.

    Attributes:
        imgs (numpy.ndarray): Input images.
        trues (numpy.ndarray): True masks.

    Usage:
        performance_plot_callback = PerformancePlotCallback(val_gen, batch)
        model.fit(..., callbacks=[performance_plot_callback])
    """
    def __init__(self, val_gen, batch):
        batch = val_gen.__getitem__(batch)
        self.imgs = batch[0]
        self.trues = batch[1]
        
    def on_epoch_end(self, epoch, logs={}):
        """
        Plot model performance at the end of each epoch.

        Args:
            epoch (int): Current epoch number.
            logs (dict): Dictionary of training metrics.
        """
        # Generate predictions
        y_preds = self.model.predict(self.imgs)
        
        # Plot input image, true mask, and predicted mask
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
    """
    Custom Keras callback to save model checkpoints at specific epochs.

    This callback saves the model checkpoint at specified epochs during training.

    Args:
        checkpoint_dir (str): Directory to save the model checkpoints.
        fname (str): Filename for the model checkpoint.
        save_epochs (list): List of epochs at which to save the model.

    Usage:
        checkpoint_callback = SaveCheckpointAtEpoch(checkpoint_dir, fname, save_epochs)
        model.fit(..., callbacks=[checkpoint_callback])
    """
    def __init__(self, checkpoint_dir, fname, save_epochs):
        super(SaveCheckpointAtEpoch, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.fname = fname
        self.save_epochs = save_epochs

    def on_epoch_end(self, epoch, logs=None):
        """
        Save model checkpoint at the end of specified epochs.

        Args:
            epoch (int): Current epoch number.
            logs (dict): Dictionary of training metrics.
        """
        # Check if the current epoch is in the list of epochs to save
        if (epoch + 1) in self.save_epochs:
            model_checkpoint_path = os.path.join(self.checkpoint_dir, 
                                                 self.fname.replace('epoch1', str(epoch+1)))
            # Save the model checkpoint
            self.model.save(model_checkpoint_path)
            print(f"""======> Model checkpoint saved at epoch {epoch+1}.
            """)