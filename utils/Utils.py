import numpy as np
import cv2


class Utils:

    @staticmethod
    def calculate_train_length(images_list, validation_length, test_length):
        """
               Calculates the length of the training set given the total number of images,
               and the lengths of the validation set and test set.

               Args:
               - images_list (list): A list of images.
               - validation_length (int): The length of the validation set.
               - test_length (int): The length of the test set.

               Returns:
               - train_length (int): The length of the training set.
               """
        return len(images_list) - validation_length - test_length

    @staticmethod
    def calculate_steps_per_epoch(train_length, batch_size):
        """
               Calculates the number of steps required to complete one epoch of training.

               Args:
               - train_length (int): The length of the training set.
               - batch_size (int): The batch size.

               Returns:
               - steps_per_epoch (int): The number of steps required to complete one epoch of training.
               """
        return train_length // batch_size

    @staticmethod
    def one_hot(a, num_classes):
        """
               Converts an array of integers into a one-hot encoded array.

               Args:
               - a (array): An array of integers.
               - num_classes (int): The number of classes.

               Returns:
               - one_hot_array (array): A one-hot encoded array.
               """
        return np.squeeze(np.eye(num_classes)[a])

    @staticmethod
    def rle_to_mask(rle: str, shape=(768, 768)):
        '''
        :param rle: run length encoded pixels as string formated
               shape: (height,width) of array to return
        :return: numpy 2D array, 1 - mask, 0 - background
        '''
        encoded_pixels = np.array(rle.split(), dtype=int)
        starts = encoded_pixels[::2] - 1
        ends = starts + encoded_pixels[1::2]
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape).T  # Needed to align to RLE direction

    @staticmethod
    def mask_to_rle(img, shape=(768, 768)) -> str:
        """
        :param img: numpy 2D array, 1 - mask, 0 - background
               shape: (height,width) dimensions of the image
        :return: run length encoded pixels as string formated
        """
        img = img.astype('float32')
        img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
        img = np.stack(np.vectorize(lambda x: 0 if x < 0.1 else 1)(img), axis=1)
        pixels = img.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)
