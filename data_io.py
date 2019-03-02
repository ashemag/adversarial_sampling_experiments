import numpy as np
import os
from adversarial_sampling_experiments.data_providers import CIFAR10
from adversarial_sampling_experiments.globals import ROOT_DIR
import pickle

class ImageDataIO(object):
    '''
    returns x, y.
        x is a batch of images, numpy array.
        x has shape (batch_size, num_channels, height, width).
        y is class label, integer encoded, numpy array.
        y has shape (batch_size,)
    '''

    DEFAULT_SEED = 20112018

    @staticmethod
    def load(filename):
        loaded = np.load(filename)
        x, y = loaded['inputs'], loaded['targets']

        # TODO: Enable cifar10 to be loaded.

        x = x.reshape(len(x),1,28,-1)
        x = x.astype(float) # correct type for imshow
        y = y.astype(int)

        return x, y

    @staticmethod
    def save_data(x, y, filename_npz):
        '''
        :param x: batch of augmented images.
            type: numpy array.
            shape: (batch_size, num_channels, height, width)
        :param y: class labels, integer encoded.
            type: numpy array,
            shape: (batch_size,)
        '''

        directory = os.path.dirname(filename_npz)
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.savez(filename_npz, inputs=x,targets=y)
        print("saved data. ", filename_npz)

    @staticmethod
    def mnist_augmented(which_set='train',which_attack='maxpgd1'):
        if which_set == 'train' and which_attack == 'maxpgd1':
            filename = os.path.join(ROOT_DIR,'data/mnist-{}-{}'.format(which_set,which_attack))
            return ImageDataIO.load(filename=filename)

    @staticmethod
    def mnist(which_set='train'):
        if which_set == 'train':
            return ImageDataIO.load(filename=os.path.join(ROOT_DIR,'data/mnist-train.npz'))
        if which_set == 'valid':
            return ImageDataIO.load(filename=os.path.join(ROOT_DIR,'data/mnist-valid.npz'))
        if which_set == 'test':
            return ImageDataIO.load(filename=os.path.join(ROOT_DIR,'data/mnist-test.npz'))

    @staticmethod
    def cifar10(which_set='train'):
        if which_set == 'train':
            return ImageDataIO.load(filename=os.path.join(ROOT_DIR, 'data/cifar10-train.npz'))
        if which_set == 'valid':
            return ImageDataIO.load(filename=os.path.join(ROOT_DIR, 'data/cifar10-valid.npz'))
        if which_set == 'test':
            return ImageDataIO.load(filename=os.path.join(ROOT_DIR, 'data/cifar10-test.npz'))

    @staticmethod
    def download_cifar10():
        x, y = ImageDataIO.cifar10_old(which_set='train')
        #     ImageDataIO.save_data(x,y,filename_npz=os.path.join(ROOT_DIR,'data/cifar10-train.npz'))




        pass

    @staticmethod
    def cifar10_old(which_set='train'):
        from adversarial_sampling_experiments.globals import ROOT_DIR
        data_dir = os.path.join(ROOT_DIR,'data')
        loaded = CIFAR10(root=data_dir,set_name=which_set,download=False)
        x = np.transpose(loaded.data,(0,3,1,2)) # (1)
        y = loaded.labels
        x = x.astype(float)  # correct type for imshow
        y = y.astype(int)

        '''
        remarks:
        (1)
        loaded.data is (batch_size,height,width,num_channels),
        x is reshaped to match (batch_size,num_channels,height,width)
        '''

        return x, y