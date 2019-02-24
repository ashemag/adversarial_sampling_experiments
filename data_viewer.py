import matplotlib.pyplot as plt
import numpy as np
from adversarial_sampling_experiments.data_subsetter import DataSubsetter

class ImageDataViewer():
    def __init__(self,data):
        self.inputs = data[0]
        self.int_targets = data[1]

        pass

    @staticmethod
    def grid(plot_dict,hspace=0.5):
        '''
        :param plot_dict: dictionary of form:
            plot_dict['label']['ax'] = ax, where ax is an Axis.
            plot_dict['label']['img'] = x, where x is an array of shape (num_channels, height, width).(single image)
        '''

        for k in plot_dict.keys():
            ax = plot_dict[k]['ax']
            img = plot_dict[k]['img']
            x_label = plot_dict[k]['x_label']
            img = np.transpose(img,(1,2,0)) # correct format for imshow
            if img.shape[2]==1:
                img = np.reshape(img,(img.shape[0],-1)) # imshow doesn't accept (height, width, 1).
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(x_label)

        if hspace is not None:
            plt.subplots_adjust(hspace=hspace)

        plt.show()


    def grid_subset_of(self, shape, label):
        '''
        displays a grid of images. see notebook for usage.
        '''
        data = (self.inputs,self.int_targets)
        subset = DataSubsetter.condition_on_label(data, labels=[label], shuffle=True)
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