import matplotlib.pyplot as plt
import numpy as np
from adversarial_sampling_experiments.experiment.utils import ModelMetrics

class ImageDataViewer():
    def __init__(self,data):
        self.inputs = data[0]
        self.int_targets = data[1]

        pass

    @staticmethod
    def grid(plot_dict):
        '''
        :param plot_dict: dictionary of form:
            plot_dict['label']['ax'] = ax, where ax is an Axis.
            plot_dict['label']['img'] = x, where x is an array of shape (1, num_channels, height, width).
        '''

        for label in plot_dict.keys():
            ax = plot_dict[label]['ax']
            img = plot_dict[label]['img']
            img = img.reshape(img,img.shape[1:])
            img = img.transpose(img,(1,2,0)) # correct format for imshow
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(label)

        plt.show()


    def grid_subset_of(self, shape, label):
        '''
        displays a grid of images. see notebook for usage.
        '''
        data = (self.inputs,self.int_targets)
        subset = DataHandler.condition_on_label(data, labels=[label], shuffle=True)
        images, labels = subset
        fig, axs = plt.subplots(shape[0],shape[1])
        axs = axs.reshape(-1,) # originally returns as 2D matrix.
        for i in range(len(axs)):
            axs[i].imshow(images[i])
            axs[i].set_xticks([])
            axs[i].set_yticks([])

        return fig, axs

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