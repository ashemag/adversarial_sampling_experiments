import matplotlib.pyplot as plt
import numpy as np
from data_subsetter import DataSubsetter
from data_io import ImageDataIO
import os
from globals import ROOT_DIR
import pickle

def test2():
    filename = os.path.join(ROOT_DIR,'results/cifar10_advers_exp2_0.01/advers_images.pickle')
    with open(filename,mode='rb') as file:
        images_dict = pickle.load(file)

    print("length image dict: ",len(images_dict.keys()))
    xx = np.zeros((0,3,32,32))
    pixel_ub = 255

    img_range = range(90,120)

    for i in img_range:
        yy = (images_dict[i]/pixel_ub).astype(np.float)
        zz = np.reshape(yy[0],(1,3,32,32))
        xx = np.vstack((xx,zz))

    labels = [i for i in img_range]

    ImageDataViewer.batch_view(xx,nrows=int(len(xx)/6),ncols=6,labels=labels,hspace=0.1,cmap=None)

    pass

def test_mnist():
    filename = os.path.join(ROOT_DIR,'results/advers_normal_mnist_simple/advers_images.pickle')
    with open(filename,mode='rb') as file:
        images_dict = pickle.load(file)

    ub = np.max(images_dict[0])
    print("ub: ",ub)
    pixel_ub = 1

    xx = np.zeros((0,1,28,28))

    num_channels = 1
    height = 28
    width = 28

    num_pictures = 20

    for i in range(num_pictures):
        yy = (images_dict[i]/pixel_ub).astype(np.float)
        zz = np.reshape(yy[0],(1,num_channels,height,width))
        xx = np.vstack((xx,zz))

    labels = [i for i in range(len(images_dict.keys()))]

    ImageDataViewer.batch_view(xx,nrows=int(num_pictures/5),ncols=5,labels=labels)

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
    def batch_view(x,nrows,ncols,labels,cmap,hspace,wspace):
        if len(x) != nrows*ncols: raise ValueError('dimension mismatch.')
       # if np.max(x[0]) > 1.05: raise ValueError('pixel values must be float between 0 and 1.')

        fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(10,10))
        axs = axs.flatten()

        # labels = ['epoch_{}'.format(i) for i in range(len(axs))]  # labels to put beneath each image.
        plot_dict = {label:{'ax':ax,'img':img,'x_label':label} for label,ax,img in zip(labels,axs,x)}
        if cmap is None:
            ImageDataViewer.grid(plot_dict,hspace=hspace,wspace=wspace)  # hspace controls spacing between images.
        else:
            ImageDataViewer.grid(plot_dict, cmap=cmap, hspace=hspace,wspace=wspace)
        plt.show()


    @staticmethod
    def grid(plot_dict,hspace,wspace,cmap=None):
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
            if cmap is not None:
                ax.imshow(img,cmap=cmap)
            else:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(x_label)

        #plt.subplots_adjust(hspace=hspace,wspace=wspace) # can be used to adjust subplots.
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
    test2()