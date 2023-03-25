class Constants:
    """
       A class to store constant variables used throughout the project.
       """

    def __init__(self):
        """
        Constructor for the Constants class. Does nothing.
        """
        pass

    RANDOM_SEED = 77
    TRAIN_DIR = 'data/airbus-ship-detection/train_v2/'
    TEST_DIR = 'data/airbus-ship-detection/test_v2/'
    CORRUPTED_IMAGES = ['6384c3e78.jpg']
    IMAGES_WITHOUT_SHIPS_NUMBER = 25000
    VALIDATION_LENGTH = 2000
    TEST_LENGTH = 2000
    BATCH_SIZE = 16
    BUFFER_SIZE = 1000
    IMG_SHAPE = (256, 256)
    NUM_CLASSES = 2
    EPOCHS = 1
    CHECKPOINT_FILEPATH = 'models/wheight/model-checkpoint'
