import numpy as np
import os
from data_providers import CIFAR10
from globals import ROOT_DIR

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
        x = x.astype(float) # correct type for imshow
        y = y.astype(int)
        # x is shape (batch_size, num_channels, height, width) = (-1,3,32,32) for cifar-10.
        return x, y

    @staticmethod
    def load_mnist(filename):
        loaded = np.load(filename)
        x, y = loaded['inputs'], loaded['targets']
        # the problem is mnist is not saved properly - so we have to reshape it.

        x = x.reshape(len(x), 1, 28, -1)
        x = x.astype(float)  # correct type for imshow
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
            return ImageDataIO.load_mnist(filename=os.path.join(ROOT_DIR,'data/mnist-train.npz'))
        if which_set == 'valid':
            return ImageDataIO.load_mnist(filename=os.path.join(ROOT_DIR,'data/mnist-valid.npz'))
        if which_set == 'test':
            return ImageDataIO.load_mnist(filename=os.path.join(ROOT_DIR,'data/mnist-test.npz'))

    @staticmethod
    def cifar10(which_set='train',normalize=False):
        x, y = None, None
        if which_set == 'train':
            x, y = ImageDataIO.load(filename=os.path.join(ROOT_DIR, 'data/cifar10-train.npz'))
        if which_set == 'valid':
            x, y = ImageDataIO.load(filename=os.path.join(ROOT_DIR, 'data/cifar10-valid.npz'))
        if which_set == 'test':
            x, y = ImageDataIO.load(filename=os.path.join(ROOT_DIR, 'data/cifar10-test.npz'))

        # import sys
        # from matplotlib import pyplot as plt
        # print(x.shape)
        # zz = np.transpose(x[0],(1, 2, 0))
        # print(zz.shape)
        # print("zz mean: ",np.mean(zz), "min: ",np.min(zz)," max: ",np.max(zz))
        # #zz = np.random.uniform(0, 1, (28, 28, 3))
        # plt.imshow(zz/255.)
        # plt.show()
        #
        # sys.exit()
        import torch
        from torchvision import transforms
        import sys

        # if normalize: torch.stack([transforms.ToTensor()(item) for item in x])  # 255 / 127.5 = 2, 1 0 -> 0, -1
        if normalize: x = x/127.5 - 1

        # x = x[:2]
        #
        #
        #
        #
        # x = torch.Tensor(x).float()
        # x_mixed = x  # torch.cat([x, grad_x_adv, x_adv_tens], dim=2)
        # x_mixed = torch.unbind(x_mixed, dim=0)
        # x_mixed = torch.cat(x_mixed, dim=2)
        # x_mixed = x_mixed.cpu()
        # print(torch.mean(x_mixed), torch.max(x_mixed), torch.min(x_mixed), torch.std(x_mixed))
        # x_mixed = transforms.ToPILImage()(x_mixed)
        # x_mixed.show()
        # sys.exit("Finished showing images")


        return x, y


    @staticmethod
    def download_cifar10(which_set='train'):
        from globals import ROOT_DIR
        data_dir = os.path.join(ROOT_DIR,'data')
        loaded = CIFAR10(root=data_dir,set_name=which_set,download=True)
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


if __name__ == '__main__':

    x, y = ImageDataIO.cifar10('train',normalize=True)