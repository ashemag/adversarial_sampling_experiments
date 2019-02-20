import matplotlib.pyplot as plt
import numpy as np

class ImageDataViewer():
    def __init__(self,data):
        self.inputs = data[0]
        self.int_targets = data[1]

        pass

    def grid(self, shape, label):
        '''
        displays a grid of images. see notebook for usage.
        '''
        data = (self.inputs,self.int_targets)
        subset = DataHandler.make_subset(data, targets=[label], shuffle=True)
        images, labels = subset
        fig, axs = plt.subplots(shape[0],shape[1])
        axs = axs.reshape(-1,) # originally returns as 2D matrix.
        for i in range(len(axs)):
            axs[i].imshow(images[i])
            axs[i].set_xticks([])
            axs[i].set_yticks([])

        return fig, axs


class DataHandler():
    DEFAULT_SEED = 20112018

    @staticmethod
    def make_subset(data, targets, shuffle=False, rng=None):
        images, labels = data
        images_subset = np.array([images[i] for i in range(len(images)) if labels[i] in targets])
        labels_subset = np.array([labels[i] for i in range(len(images)) if labels[i] in targets])
        out = (images_subset,labels_subset)
        if shuffle:
            out = DataHandler.shuffle(out,rng)
        return out

    @staticmethod
    def shuffle(data,rng=None):
        if rng is None:
            rng = np.random.RandomState(DataHandler.DEFAULT_SEED)
        images, labels = data
        perm = rng.permutation(len(images))
        shuffled = (images[perm], labels[perm])
        return shuffled


if __name__ == '__main__':
    # import os
    # from adversarial_sampling_experiments.data_providers import CIFAR10
    # from adversarial_sampling_experiments import globals
    #
    # data_dir = os.path.join(globals.ROOT_DIR, 'data')
    # cifar10 = CIFAR10(root=data_dir, set_name='val', download=False)
    # images = cifar10.data
    # labels = cifar10.labels
    #
    # viewer = ImageDataViewer((images, labels))
    # fig, axs = viewer.grid(shape=(4, 4), label=0)
    #
    # plt.show()
    pass