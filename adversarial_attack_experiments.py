"""
Module for running adversarial attack experiments
"""
import argparse
from copy import copy

from PIL import Image
from torch.utils.data import DataLoader
from data_providers import MinorityDataLoader
from torchvision import transforms
import numpy as np
from minority_class_experiments import unpickle
from models.densenet import DenseNet121
import torch
from torch import optim
import os
from globals import ROOT_DIR
import sys
from attacks.advers_attacks import LInfProjectedGradientAttack
from models.base import Logger
from attacks.advers_attacks import RotateAttack
from data_providers import CIFAR10
import matplotlib
#matplotlib.use('TkAgg')

logger = Logger(stream=sys.stderr, disable=False)
LEARNING_RATE = .1
WEIGHT_DECAY = 1e-4
MOMENTUM = .9
NUM_EPOCHS = 120
LOGS = False

def get_transform(set_name, BASIC_FLAG=True):
    """
    Standard is different transforms for train/valid
    :param set_name:
    :return:
    """
    if BASIC_FLAG:
        return transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if set_name == 'train':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


def get_args():
    parser = argparse.ArgumentParser(description='Minority class data experiment.')
    parser.add_argument('--cpu', type=bool, default=False)
    args = parser.parse_args()
    arg_str = [(str(key), str(value)) for (key, value) in vars(args).items()]
    print("Arguments: {}".format(arg_str))
    return args


def cifar10_experiment(minority_class_idx, minority_percentage, results_dir, loss_based_attack=True, rotated_attack=False, epsilon=40 / 255):
    args = get_args()
    percentages = [1. for _ in range(10)]
    percentages_mod = copy(percentages)
    percentages_mod[minority_class_idx] = minority_percentage

    train_set = CIFAR10(root='data', transform=get_transform('train'), download=True, set_name='train',
                        percentages_list=percentages_mod)

    valid_set = CIFAR10(root='data', transform=get_transform('valid'), download=True, set_name='val',
                        percentages_list=percentages)

    test_set = CIFAR10(root='data', transform=get_transform('train'), download=True, set_name='test',
                       percentages_list=percentages)

    train_data = MinorityDataLoader(train_set, batch_size=64, shuffle=True, num_workers=4, minority_class_idx=3)
    valid_data = torch.utils.data.DataLoader(valid_set, batch_size=64, shuffle=True, num_workers=4)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=4)

    model = DenseNet121()
    if not args.cpu:
        model = model.to(model.device)

    if loss_based_attack:
        attack = LInfProjectedGradientAttack(
            model=model,
            steps=1,
            alpha=1,
            epsilon=epsilon,
            rand=True,
            targeted=False
        )
    elif rotated_attack:
        attack = RotateAttack(model=model)
    else:
        raise Exception('You did not specify an attack.')

    if LOGS:
        for batch in train_data:
            (x_maj_batch, y_maj_batch, x_min_batch, y_min_batch) = batch
            print("Batch sizes are {} majority and {} minority".format(len(x_maj_batch), len(x_min_batch)))
            img = x_min_batch[0].detach().numpy()
            img = np.transpose(img, (1, 2, 0))
            result = Image.fromarray((img * 255).astype(np.uint8))
            result.save('data/attacks/standard_test.png')

            x_min_batch_adv = attack(x_min_batch, y_min_batch)
            img = x_min_batch_adv[0].numpy()
            img = np.transpose(img, (1, 2, 0))
            result = Image.fromarray((img * 255).astype(np.uint8))
            result.save('data/attacks/attack_test.png')

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0.0001)

    model.train_evaluate(
        train_set=train_data,
        valid_set=valid_data,
        test_set=test_data,
        attack=attack,
        num_epochs=NUM_EPOCHS,
        optimizer=optimizer,
        results_dir=results_dir,
        scheduler=scheduler,
        minority_class_idx=minority_class_idx,
    )


def get_index_mapping():
    d = unpickle('data/cifar-10-batches-py/batches.meta')
    labels = d[b'label_names']
    return {index: value.decode('ascii') for index, value in enumerate(labels)}


if __name__ == '__main__':
    minority_percentage = 1 #class, X * 100 = %
    name_exp = 'loss-based attack'
    minority_class_index = 3
    index_mapping = get_index_mapping()

    print("RUNNING EXPERIMENT: {} for minority class {}".format(name_exp, index_mapping[minority_class_index]))

    # SET LOGS
    results_dir = os.path.join(ROOT_DIR,'results/{}_{}'.format(name_exp, minority_percentage))

    cifar10_experiment(
        minority_class_idx=minority_class_index,
        results_dir=results_dir,
        minority_percentage=minority_percentage,
        rotated_attack=False,
        loss_based_attack=True,
        epsilon=4/255
    )

