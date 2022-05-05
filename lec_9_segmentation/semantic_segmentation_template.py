import os
import sys

import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, concatenate
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.utils import plot_model


class Dataset():
    def __init__(self, dataset_folder, target_size, batch_size):
        """
        Parameters
        ----------
        dataset_folder (str): The folder where the dataset is located
        target_size (tuple): The height, width of the target iamge
        batch_size (int): The batch sized there shoud be used
        """
        self.root_dataset_folder = dataset_folder
        self.mask_correction_threshold = 0.1
        self.batch_size = batch_size
        self.target_size = target_size
        self.clip_limit = 0.03

    def make_generator(self, dataset_folder, img_folder, aug_dict, seed=1, color_mode="grayscale"):
        """
        Creates a image generator

        Parameters
        ----------
        dataset_folder (str): The folder where the dataset is located
        img_folder (str): The folder in the dataset_folder there has the images there should be used
        aug_dict (dict): Augments for the ImageDataGenerator
        seed (int): The seed for the generator
        color_mode (str): What color mode to load images in

        Returns
        -------
        data_gen (DirectoryIterator): The generator
        """
        data_gen = ImageDataGenerator(**aug_dict)
        data_gen = data_gen.flow_from_directory(
            dataset_folder,
            classes=[img_folder],
            class_mode=None,
            color_mode=color_mode,
            target_size=self.target_size,
            batch_size=self.batch_size,
            seed=seed)
        return data_gen

    def get_image_mask_generator(self, split, data_gen_args):
        """
        Creates a image+mask generator

        Parameters
        ----------
        split (str): What split the generator should be made for
        data_gen_args (dict): Augments for the ImageDataGenerator

        Returns
        -------
        gen (generator): The generator
        """
        dataset_folder = os.path.join(self.root_dataset_folder, split)

        # Create the generators for the images and the masks
        img_gen = self.make_generator(dataset_folder, 'images', data_gen_args, color_mode="grayscale")
        mask_gen = self.make_generator(dataset_folder, 'vessel', data_gen_args, color_mode="grayscale")
        data_gen = zip(img_gen, mask_gen)

        # Create a new generators there combines the two others
        def gen():
            for img, mask in data_gen:
                # Correct for mask values changes do to augmentations
                mask[mask > self.mask_correction_threshold] = 1
                mask[mask <= self.mask_correction_threshold] = 0
                yield img, mask

        return gen

    def get_train_set(self):
        """
        Creates the training image+mask generator

        Returns
        -------
        gen (generator): The generator
        """
        data_gen_args = dict(rotation_range=0.2,
                             width_shift_range=0.05,
                             height_shift_range=0.05,
                             shear_range=0.05,
                             zoom_range=0.05,
                             rescale=1.0 / 255,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='constant',
                             cval=0)

        gen = self.get_image_mask_generator("training", data_gen_args)
        return gen()

    def get_val_set(self):
        """
        Creates the validation image+mask generator

        Returns
        -------
        gen (generator): The generator
        """
        data_gen_args = dict(rescale=1.0 / 255,
                             horizontal_flip=True,
                             vertical_flip=True)

        gen = self.get_image_mask_generator("validation", data_gen_args)

        return gen()

    def get_test_set(self):
        """
        Creates the test image+mask generator

        Returns
        -------
        gen (generator): The generator
        """
        dataset_folder = os.path.join(self.root_dataset_folder, "test", "images")
        for img in os.listdir(dataset_folder):
            img = cv2.imread(os.path.join(dataset_folder, img), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.target_size)
            img = img / 255.
            yield np.array([img])


class UNet():
    @staticmethod
    def conv_block(net, n_filters, use_batch_norm=False):
        """
        Creates a conv block for a UNet model

        Parameters
        ----------
        net (KerasTensor): The input layer for the conv block
        n_filters (int): The number of filters
        use_batch_norm (bool): If True there will be used batch normalization

        Returns
        -------
        net (KerasTensor): The output layer for the conv block
        """
        net = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(net)
        if use_batch_norm:
            net = BatchNormalization()(net)
        net = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(net)
        if use_batch_norm:
            net = BatchNormalization()(net)
        return net

    @staticmethod
    def upsample_block(net, n_filters, upsample_size=(2, 2), use_batch_norm=False):
        """
        Creates a upsample block for a UNet model

        Parameters
        ----------
        net (KerasTensor): The input layer for the conv block
        n_filters (int): The number of filters
        upsample_size (Int, or tuple of 2 integers): The upsampling factors for rows and columns.
        use_batch_norm (bool): If True there will be used batch normalization

        Returns
        -------
        net (KerasTensor): The output layer for the conv block
        """
        net = UpSampling2D(upsample_size)(net)
        net = Conv2D(n_filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(net)
        if use_batch_norm:
            net = BatchNormalization()(net)
        return net

    def build_model(self, input_size, n_layers, n_filters, use_batch_norm):
        """
        Builds the UNet model

        Parameters
        ----------
        input_size (tuple): The inputs size of the image for the model
        n_layers (int): The number of layers leading to the middle and from the middle
        n_filters (itn): The number of filters in the first layer
        use_batch_norm (bool): If True there will be used batch normalization
        """
        inputs = Input(input_size)
        net = inputs
        down_layers = []

        # Create the conv blocks
        for _ in range(n_layers):
            net = self.conv_block(net, n_filters, use_batch_norm)
            print(net.get_shape())
            down_layers.append(net)
            net = MaxPooling2D((2, 2), strides=2)(net)
            n_filters *= 2

        # Complete the network
        # Start with a dropout layer followed by a conv block to create the bottom of the U
        # For each layer in down_layers, concatenate it with the upsampled output from the previous layer, followed by a conv block
        # Remember to half the amount of filters per upsampling
        # Finish off with a sigmoid activated Conv layer with one (1x1) filter
        # Create a model = Model(inputs=inputs, outputs=sigmoid)
        # Show the architecture with model.summary()

    def train(self, train_set, val_set, steps_per_epoch, epochs, lr):
        """
        Trains the model

        Parameters
        ----------
        train_set (generator): A image+mask generator for the training data
        val_set (generator): A image+mask generator for the validation data
        steps_per_epoch (int): The number of steps per epoch
        epochs (int): The number of epochs
        lr (float): The learning rate
        """
        # Compile the model using the Adam optimizer and a binary_crossentropy loss function
        # Create a learning rate scheduler callback (use learning rate 0.0001 for the first 3 epochs, 0.00005 for the rest)
        # Create a model checkpoint saving callback. Save only the best model based on accuracy
        # Fit the model using the train set. Remember to pass the callbacks
        pass

    def predict(self, test_set):
        """
        Predict on the given data

        Parameters
        ----------
        test_set (generator): A image+mask generator for the test data
        """
        # Use model.predict to segment each image in the test set
        # Threshold the prediction around 0.5
        # Show the input image and thresholded prediction (remember to remove excess axes)
        pass


if __name__ == "__main__":
    dataset_folder = '../data/RetinaExtraction'

    input_size =  ( , , 1)  # The input size for the model
    n_layers = # The number of layers leading to the middle and from the middle
    n_filters = # The first layers number of filters
    use_batch_norm = # Use batch normalization or not

    batch_size = # The batch size used in the training
    epochs = # The number of epochs to train over
    steps_per_epoch = # The number of training batches to preform for each epoch
    lr = # The learning rate to start with

    dataset = Dataset(dataset_folder, batch_size=batch_size, target_size=input_size[:2])

    train_set = dataset.get_train_set()
    val_set = dataset.get_val_set()

    print('Visualize augmented examples.')
    print('Press ESC to continue.')
    for image, label in train_set:
        cv2.imshow('image', cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR))
        cv2.imshow('label', label[0])
        key = cv2.waitKey(0)
        if key == 27:
            break
    cv2.destroyAllWindows()

    unet = UNet()
    unet.build_model(input_size, n_layers, n_filters, use_batch_norm)
    unet.train(train_set, val_set, epochs=epochs, steps_per_epoch=steps_per_epoch, lr=lr)

    test_set = dataset.get_test_set()
    unet.predict(test_set)
