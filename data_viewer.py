import matplotlib.pyplot as plt
import numpy as np
from data_subsetter import DataSubsetter
from data_io import ImageDataIO
import os
from globals import ROOT_DIR
import pickle

def test2():
    filename = os.path.join(ROOT_DIR,'ExperimentResults/advers_images_cifar10.pickle')
    with open(filename,mode='rb') as file:
        images_dict = pickle.load(file)

    xx = np.zeros((0,3,32,32))

    for i in range(30,60):
        yy = (images_dict[i]/255).astype(np.float)
        zz = np.reshape(yy[0],(1,3,32,32))
        xx = np.vstack((xx,zz))

    labels = [i for i in range(30,60)]

    ImageDataViewer.batch_view(xx,nrows=int(len(xx)/6),ncols=6,labels=labels)


    pass


def test():
    '''
    how to use ImageDataViewer.grid:
    the example below will create a 3 by 2 grid of images.
    '''

    fig, axs = plt.subplots(nrows=3,ncols=2)
    axs = axs.flatten() # must be flattened!
    labels = ['label_{}'.format(i) for i in range(len(axs))] # labels to put beneath each image.
    x, y = ImageDataIO.mnist('train') # x is numpy array of shape (batch_size,num_channels,height,width).
    x = x[:6] # first 6 images.

    plot_dict = {label:{'ax':ax,'img':img,'x_label':label} for label,ax,img in zip(labels,axs,x)} # note: images are numpy arrays.
    ImageDataViewer.grid(plot_dict,hspace=0.5) # hspace controls spacing between images.
    plt.show()

class ImageDataViewer():
    def __init__(self,data):
        self.inputs = data[0]
        self.int_targets = data[1]

        pass

    @staticmethod
    def batch_view(x,nrows,ncols,labels):
        if len(x) != nrows*ncols: raise ValueError('dimension mismatch.')

        fig, axs = plt.subplots(nrows=nrows,ncols=ncols)
        axs = axs.flatten()

        # labels = ['epoch_{}'.format(i) for i in range(len(axs))]  # labels to put beneath each image.
        plot_dict = {label:{'ax':ax,'img':img,'x_label':label} for label,ax,img in zip(labels,axs,x)}
        ImageDataViewer.grid(plot_dict, hspace=0.5)  # hspace controls spacing between images.
        plt.show()


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
    test2()
    pass