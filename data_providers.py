# -*- coding: utf-8 -*-
"""Data providers.

This module provides classes for loading datasets and iterating over batches of
data points.
"""
from __future__ import print_function
import torch
from torch.utils.data.dataloader import default_collate, _DataLoaderIter, DataLoader
from PIL import Image
import os
import os.path
import numpy as np
import sys
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
import globals
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

os.environ['MLP_DATA_DIR'] = os.path.join(globals.ROOT_DIR,'data')
DEFAULT_SEED = 20112018

class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, set_name, percentages_list,
                 transform=None, target_transform=None,
                 download=False, max_num_samples=-1):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.set_name = set_name  # training set or test set
        self.max_num_batches = max_num_samples

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.percentages_list = percentages_list

        # now load the picked numpy arrays
        rng = np.random.RandomState(seed=0)

        train_sample_idx = rng.choice(a=[i for i in range(50000)], size=47500, replace=False)
        val_sample_idx = [i for i in range(50000) if i not in train_sample_idx]

        if self.set_name is 'train':
            self.data = []
            self.labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.labels += entry['labels']
                else:
                    self.labels += entry['fine_labels']
                fo.close()

            self.data = np.concatenate(self.data)

            self.data = self.data.reshape((50000, 3, 32, 32))
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
            self.data = self.data[train_sample_idx]
            self.labels = np.array(self.labels)[train_sample_idx]

        elif self.set_name is 'val':
            self.data = []
            self.labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.labels += entry['labels']
                else:
                    self.labels += entry['fine_labels']
                fo.close()

            self.data = np.concatenate(self.data)
            self.data = self.data.reshape((50000, 3, 32, 32))
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
            self.data = self.data[val_sample_idx]
            self.labels = np.array(self.labels)[val_sample_idx]

        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.data = entry['data']
            if 'labels' in entry:
                self.labels = entry['labels']
            else:
                self.labels = entry['fine_labels']
            fo.close()
            self.data = self.data.reshape((10000, 3, 32, 32))
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
            self.labels = np.array(self.labels)

        self.data_dict = {}
        for key, value in zip(self.labels, self.data):
            if key not in self.data_dict:
                self.data_dict[key] = [value]
            else:
                self.data_dict[key].append(value)

        data_dict_sorted = {}
        for idx, key in enumerate(sorted(self.data_dict.keys())):
            self.data_dict[key] = np.array(self.data_dict[key])
            class_length = self.data_dict[key].shape[0]
            data_dict_sorted[key] = self.data_dict[key][:int(self.percentages_list[idx] * class_length)]

        self.label_to_class_idx = {label: class_idx for class_idx, label in enumerate(data_dict_sorted.keys())}

        self.data_dict = data_dict_sorted
        for key, value in data_dict_sorted.items():
            print(key, value.shape[0])

        self.data_length = len(self.data)

        if self.max_num_batches != -1:
            self.data_length = self.max_num_batches

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        rng = np.random.RandomState(index + DEFAULT_SEED)
        # img, target = self.data[index], self.labels[index]

        selected_class = rng.choice(list(self.data_dict.keys()), 1, replace=False)[0]

        selected_sample_idx = rng.choice(self.data_dict[selected_class].shape[0], 1, replace=False)

        selected_sample_x = self.data_dict[selected_class][selected_sample_idx]

        selected_sample_y = self.label_to_class_idx[selected_class]



        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = Image.fromarray(selected_sample_x[0])
        target = selected_sample_y

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.data_length

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print("{} set: Files already downloaded and verified".format(self.set_name))
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.set_name
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            inputs, label = dataset[idx][0], dataset[idx][1]
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[dataset[idx][1]] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
