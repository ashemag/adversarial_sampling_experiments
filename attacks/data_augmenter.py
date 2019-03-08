import numpy as np
from tqdm import tqdm
import os
import sys

class DataAugmenter(object):
    @staticmethod
    def advers_attack(x, y, attack, disable_progress=False):
        '''
        :param x: batch of images to be augmented by adversarial attacks.
            type: numpy array.
            shape: (batch_size, num_channels, height, width)
        :param y: batch of labels. integer encoded.
            type: numpy array, int8.
            shape: (batch_size,)
        :param attack: adverarial attack to apply.
            type: attack object e.g. LInfProjectedGradientAttack in advers_attacks.py
        '''

        x_adv = np.zeros_like(x)  # (batch_size, img_height, img_width)

        for i in tqdm(range(len(x)), disable=disable_progress,file=sys.stderr):
            xx = np.reshape(x[i],(1,x[i].shape[0],x[i].shape[1],x[i].shape[2]))
            x_adv[i] = attack(xx,y[i])

        return x_adv

    @staticmethod
    def save_data(x_aug, y, filename_npz):
        '''
        :param x_aug: batch of augmented images.
            type: numpy array.
            shape: (batch_size, num_channels, height, width)
        :param y: class labels, integer encoded.
            type: numpy array,
            shape: (batch_size,)
        '''

        directory = os.path.dirname(filename_npz)
        if not os.path.exists(directory):
            os.makedirs(directory)
        data_to_save = {'inputs': x_aug, 'targets': y}
        np.savez(filename_npz, data_to_save)
        print("saved data. ", filename_npz)

    @staticmethod
    def load_data(filename):
        loaded = np.load(filename)
        x, y = loaded['inputs'], loaded['targets']

        return x, y