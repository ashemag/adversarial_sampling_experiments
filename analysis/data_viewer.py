import matplotlib.pyplot as plt
import numpy as np
from adversarial_sampling_experiments.experiment.utils import ModelMetrics

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
    def n_best():
        '''
        if 0 then returns the best prediction, if 1 the second best prediction.
        :return:
        '''

        # possibly use this function to plot bar-chart of what each class top other class being misclassified as.

        pass

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

    @staticmethod
    def better_than(model,data,lb,metric='accuracy'):
        x, y = data
        probs, y_pred = ModelMetrics.confidence(model,x)

        x_out, y_out = [],[]
        probs_out, y_pred_out = [],[]
        for i in range(len(x)):
            if probs[i] >= lb:
                x_out.append(x[i])
                y_out.append(y[i])
                y_pred_out.append(y_pred[i])
                probs_out.append(probs[i])

        x_out, y_out = np.array(x_out), np.array(y_out)
        data_out = (x_out,y_out)

        return data_out, probs_out, y_pred_out

    @staticmethod
    def worse_than(model,ub,metric='accuracy'):

        pass

    @staticmethod
    def between(model,lb,ub,metric='accuracy'):

        pass



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