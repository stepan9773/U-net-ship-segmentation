import tensorflow as tf
import pandas as pd
from utils.Constants import Constants
from utils.Utils import Utils
import cv2
import numpy as np


class PrepData():
    def __init__(self, images_list, image_segmentation):
        """Initialize PrepData object.

                Args:
                    images_list: A tf.data.Dataset object containing the file paths of the input images.
                    image_segmentation: A Pandas DataFrame object containing the image segmentation masks.
                """
        self.images_list = images_list
        self.image_segmentation = image_segmentation
        self.train_images = self.get_train_images()

    def _load_train_image(self, tensor):
        """Load a single train image and its corresponding mask.

                Args:
                    tensor: A tensor representing the file path of the input image.

                Returns:
                    A tuple of 3 elements:
                    - A tensor of the input image
                    - A tensor of the segmentation mask
                    - A tensor of the sample weights
                """
        path = tf.get_static_value(tensor).decode("utf-8")

        image_id = path.split('/')[-1]
        input_image = cv2.imread(path)
        input_image = tf.image.resize(input_image, Constants.IMG_SHAPE)
        input_image = tf.cast(input_image, tf.float32) / 255.0

        encoded_mask = self.image_segmentation[self.image_segmentation['ImageId'] == image_id].iloc[0]['EncodedPixels']
        input_mask = np.zeros(Constants.IMG_SHAPE + (1,), dtype=np.int8)
        if not pd.isna(encoded_mask):
            input_mask = Utils.rle_to_mask(encoded_mask)
            input_mask = cv2.resize(input_mask, Constants.IMG_SHAPE, interpolation=cv2.INTER_AREA)
            input_mask = np.expand_dims(input_mask, axis=2)
        one_hot_segmentation_mask = Utils.one_hot(input_mask, Constants.NUM_CLASSES)
        input_mask_tensor = tf.convert_to_tensor(one_hot_segmentation_mask, dtype=tf.float32)

        class_weights = tf.constant([0.0005, 0.9995], tf.float32)
        sample_weights = tf.gather(class_weights, indices=tf.cast(input_mask_tensor, tf.int32),
                                   name='cast_sample_weights')

        return input_image, input_mask_tensor, sample_weights

    def get_train_images(self):
        """
                Loads the training images using the _load_train_image function.

                Returns:
                tf.data.Dataset: A dataset containing the input images, their corresponding masks and the sample weights.
                """
        self.train_images = self.images_list.map(
            lambda x: tf.py_function(self._load_train_image, [x], [tf.float32, tf.float32]),
            num_parallel_calls=tf.data.AUTOTUNE)
        return self.train_images

    def get_test_dataset(self):
        """
                Creates a test dataset from the loaded training images.

                Returns:
                tf.data.Dataset: A dataset containing the test input images, their corresponding masks and the sample weights.
                """
        self.get_train_images()
        self.test_dataset = self.train_images.skip(Constants.VALIDATION_LENGTH).take(Constants.TEST_LENGTH)
        return self.test_dataset

    def get_validation_dataset(self):
        """
                Creates a validation dataset from the loaded training images.

                Returns:
                tf.data.Dataset: A dataset containing the validation input images, their corresponding masks and the sample weights.
                """
        self.validation_dataset = self.train_images.take(Constants.VALIDATION_LENGTH)
        return self.validation_dataset

    def get_train_dataset(self):
        """
                Creates a train dataset from the loaded training images.

                Returns:
                tf.data.Dataset: A dataset
                """

        self.train_dataset = self.train_images.skip(Constants.VALIDATION_LENGTH + Constants.TEST_LENGTH)
        return self.train_dataset
