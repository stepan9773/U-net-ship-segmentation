import tensorflow as tf
import keras.backend as K


class Metric():
    """
    Class for defining evaluation metrics for a model.
    """

    @staticmethod
    def dice(targets, inputs, smooth=1e-6):
        """
            Calculates the Dice Coefficient metric for segmentation tasks.

            Args:
                targets (tensor): Ground truth segmentation mask.
                inputs (tensor): Predicted segmentation mask.
                smooth (float): Smoothing factor to avoid division by zero.

            Returns:
                dice (tensor): Dice Coefficient score between the predicted and ground truth masks.
            """
        axis = [1, 2, 3]
        intersection = K.sum(targets * inputs, axis=axis)
        dice = (2 * intersection + smooth) / (K.sum(targets, axis=axis) + K.sum(inputs, axis=axis) + smooth)
        return dice

    @staticmethod
    def bce_loss(targets, inputs, smooth=1e-6):
        """
            Calculates the Binary Cross-Entropy loss for segmentation tasks.

            Args:
                targets (tensor): Ground truth segmentation mask.
                inputs (tensor): Predicted segmentation mask.
                smooth (float): Smoothing factor to avoid division by zero.

            Returns:
                loss (tensor): Binary Cross-Entropy loss between the predicted and ground truth masks.
            """
        axis = [1, 2, 3]
        loss = K.sum(targets * tf.math.log(inputs + smooth) + (1 - targets) * tf.math.log(1 - inputs + smooth),
                     axis=axis)
        return - loss

    @staticmethod
    def bce_dice_loss(targets, inputs):
        """
            Calculates the combined loss function of Binary Cross-Entropy and Dice Coefficient for segmentation tasks.

            Args:
                targets (tensor): Ground truth segmentation mask.
                inputs (tensor): Predicted segmentation mask.

            Returns:
                bce_dice_loss (tensor): Combined loss function of Binary Cross-Entropy and Dice Coefficient between the predicted and ground truth masks.
            """
        return Metric.bce_loss(targets, inputs) - tf.math.log(Metric.dice(targets, inputs))
