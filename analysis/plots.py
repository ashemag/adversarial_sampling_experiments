import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from sklearn.decomposition import PCA
import torch
from torch.nn import functional as F
import numpy as np
from adversarial_sampling_experiments.models.simple_fnn import FeedForwardNetwork
from adversarial_sampling_experiments import data_providers
from adversarial_sampling_experiments.attacks.test import *

def test_how_often():
    print("ENTERED")

    fig, axs = plt.subplots(3,3)
    axs = axs.reshape(-1,)
    target = 0
    labels = [str(i) for i in range(10)].remove(str(target))

    x, y = ImageDataGetter.mnist(
        filename=os.path.join(globals.ROOT_DIR, 'data/mnist-train.npz')
    )

    model = FeedForwardNetwork(img_shape=(1, 28, 28), num_classes=10)
    model.load_model(
        model_path=os.path.join(globals.ROOT_DIR,'saved_models/simple_fnn/model_epoch_49')
    )

    Plotter.how_often_second_best(x, y, model, target, labels, axs=axs)



test_how_often()


class Plotter():
    cifar10_labels = {
        0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship',
        9: 'truck'
    }

    mnist_labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
    mnist_labels = {1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
    #mnist_labels = {0: '0', 1: '1'}
    #mnist_labels = {0:'0'}

    @staticmethod
    def init_data_dict(labels,type='plot'):
        data_dict = {}
        for l in labels:
            if type=='plot':
                data_dict[l] = {'x': None,'y': None}
            if type=='hist':
                data_dict[l] = {'y': None}

        return data_dict

    @staticmethod
    def how_often_second_best(x,y,model,target,labels,axs=None):
        x = torch.Tensor(x).float()
        fc = model(x)
        sm = F.softmax(fc, dim=1)  # (batch_size, num_classes)
        sm = sm.data.numpy()
        labels = Plotter.mnist_labels
        data_dict = Plotter.init_data_dict(labels, type='bar')

        # condition on everytime target was predicted.
        # then for each label count how many times it was predicted second or first.
        # create a bar chart for this. (multi colored even - one for first one for second).

        from collections import defaultdict
        label_counts = defaultdict({lambda:0})

        target_subset = [sm[i] for i in range(len(sm)) if y[i] == target]

        for label in labels:
            # compare label against target.
            for i in range(len(target_subset)):
                sorted = np.sort(target_subset[i])
                second_best_val = sorted[-2]
                if sm[i][label] >= second_best_val:
                    label_counts[label] += 1

        x_vals = [x for x in range(len(labels))]
        for i, label in enumerate(label_counts.keys()):
            counts = label_counts[label]
            axs[i].bar(x_vals, height=counts, align='center', alpha=0.5)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_xlabel('label {}'.format(label))
            axs[i].set_ylabel('Count')

        plt.show()

    @staticmethod
    def compare(x,y,model,target,ax=None):
        '''
        :param x: array.
        :param y: array. integer encoded.
        :param model:
        :param target:
        '''
        x = torch.Tensor(x).float()
        fc = model(x)
        sm = F.softmax(fc,dim=1) # (batch_size, num_classes)
        sm = sm.data.numpy()
        labels = Plotter.mnist_labels
        data_dict = Plotter.init_data_dict(labels,type='hist')

        for l in labels:
            tgt_probs = [sm[i][target] for i in range(len(sm)) if y[i]==l] # (1)
            data_dict[l]['y'] = tgt_probs

        if ax is None:
            ax = plt.subplots(1, 1)

        for l in data_dict.keys():
            yy = data_dict[l]['y']
            ax.hist(yy, bins=100, alpha=0.5, label=labels[l])

        ax.set_xlim([0,1])
        ax.legend(loc='upper right')

        '''
        (1) tgt_prob is an array of probability assigned to 'target' given that the true class is 'label'.
        '''

        #plt.show()

        return ax

    @staticmethod
    def show_clusters(x,y,model):

        def pca_project(x,model):
            pca = PCA(n_components=2)
            fc = model(x) # final layer fully connected output.
            sm = F.softmax(fc)
            sm = sm.data.numpy()  # (batch_size, num_classes)
            sm_proj = pca.fit_transform(sm) # projected softmax activations.

            return sm_proj

        sm_proj = pca_project(x,model)
        labels = Plotter.mnist_labels
        data_dict = Plotter.init_data_dict(labels)

        for i in range(len(sm_proj)):
            label = labels[y[i]] # recall y is integer encoded.
            v0 = sm_proj[i][0]
            v1 = sm_proj[i][1]
            data_dict[label]['x'].append(v0)
            data_dict[label]['y'].append(v1)

        Plotter.multiclass_scatter(data_dict,cmap=Colormap('tab20'))


    @staticmethod
    def multiclass_scatter(data_dict,cmap=Colormap('tab20')):
        fig, ax = plt.subplots(1, 1)
        groups = []
        labels = []

        for label in data_dict.keys():
            x = data_dict[label]['x']
            y = data_dict[label]['y']
            groups.append(ax.scatter(x, y, cmap=cmap))
            labels.append(label)

        ax.legend(groups, labels)

        return fig, ax

    @staticmethod
    def multiclass_lineplot(data_dict):
        plt.figure()

        for metric in data_dict.keys():
            xx = data_dict[metric]['x_val']
            yy = data_dict[metric]['y_val']
            plt.plot(xx, yy, label=data_dict[metric]['label'])

        plt.legend()
        plt.show()

    @staticmethod
    def metric_epoch_plot(*metrics):
        '''
        :param metrics: 3-tuple (desired, label_names, filename).
        (1) "desired" is a list of strings e.g. ['acc', ..., 'loss'] corresponding to header names of the file
        you want to retrieve data from.
        (2) "label_names" is also a list of strings ['accuracy',...,'train_loss'] of how you want data on your plot to
        be labelled as.
        (3) "filename" is the path to the file you want to retrieve data from.

        :return:
        line multi-class line plot of performance metrics over epochs. can for e.g. be used to plot a learning curve.
        '''

        for (target_names,labels,filename) in metrics:
            metrics_dict = {label: {} for label in labels}

            with open(filename, 'r') as file:
                source_names = file.__next__().split()
                if bool(set(source_names) & set(target_names)):  # if any items in common
                    tgt2src = {i : source_names.index(target_name) for i,target_name in enumerate(target_names)}
                    for row_idx,line in enumerate(file):
                        row = line.split()
                        for tgt,src in tgt2src.items():
                            target_value = float(row[src])
                            metrics_dict[tgt]['x_val'].append(row_idx)
                            metrics_dict[tgt]['y_val'].append(target_value)
                            metrics_dict[tgt]['label'] = labels[tgt]

        Plotter.multiclass_lineplot(metrics_dict)


def test_compare():
    import os
    from adversarial_sampling_experiments import globals

    rng = np.random.RandomState(seed=9112018)
    data_provider = data_providers.MNISTDataProvider('train', batch_size=10000, rng=rng, max_num_batches=1)
    load_model_from = os.path.join(globals.ROOT_DIR, 'saved_models/simple_fnn/model_epoch_49')
    model = FeedForwardNetwork(img_shape=(28, 28), num_channels=1, h_out=100, num_classes=10)
    model.load_model(load_model_from)

    x, y = data_provider.__next__() # x, y. y is one-hot. x is (100,784). 28*28 = 784
    y_int = np.argmax(y,axis=1)

    print("shape y_int: ",y_int.shape)

    fig, axs = plt.subplots(1,5)

    for i,ax in enumerate(axs):
        Plotter.compare(x, y_int, model, target=i, ax=ax)


    print("shape axs: ",axs.shape)


    plt.show()

# test_compare()