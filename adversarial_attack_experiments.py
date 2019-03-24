from copy import copy
from torch.utils.data import DataLoader
from data_providers import MinorityDataLoader
from torchvision import transforms
from models.densenet import DenseNet121
import torch
from torch import optim
import os
from globals import ROOT_DIR
import sys
from attacks.advers_attacks import LInfProjectedGradientAttack
from models.base import Logger
from attacks.advers_attacks import RotateAttack

logger = Logger(stream = sys.stderr,disable= False)
LEARNING_RATE = .1
WEIGHT_DECAY = 1e-4
MOMENTUM = .9
NUM_EPOCHS = 120
MINORITY_CLASS = 3


def get_transform(set_name):
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


def cifar10_experiment(minority_percentage,results_dir, advers=False, rotated_attack=False, epsilon=40 / 255):
    from data_providers import CIFAR10

    percentages = [1. for i in range(10)]
    percentages_mod = copy(percentages)
    percentages_mod[MINORITY_CLASS] = minority_percentage
    train_set = CIFAR10(root='data', transform = get_transform('train'), download=True, set_name='train',
                        percentages_list=percentages_mod)

    valid_set = CIFAR10(root='data', transform = get_transform('valid'), download=True, set_name='val',
                        percentages_list=percentages)

    test_set = CIFAR10(root='data', transform = get_transform('valid'), download=True, set_name='test', percentages_list=percentages)

    print("Train", len(train_set))
    print("Valid", len(valid_set))
    print("Test", len(test_set))

    test_data = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=4)
    train_data = MinorityDataLoader(train_set, batch_size=64, shuffle=True, num_workers=4, minority_class_idx=3)
    valid_data = torch.utils.data.DataLoader(valid_set, batch_size=64, shuffle=True, num_workers=4)

    model = DenseNet121()
    model.use_gpu(gpu_ids='0')

    if advers:
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
        raise Exception('you did not specify an attack.')

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True,weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0.0001)

    model.train_evaluate(
        train_sampler=train_data,
        valid_full=valid_data,
        test_full=test_data,
        attack=attack,
        num_epochs=NUM_EPOCHS,
        optimizer=optimizer,
        results_dir=results_dir,
        scheduler=scheduler
    )


if __name__ == '__main__':
    minority_percentage = 1 #class, X * 100 = %
    name_exp = 'attack_rotate'
    rotated_attack = True
    loss_based_attack = False

    epsilon = 40/255
    print("RUNNING EXPERIMENET: {0}, {1}".format(name_exp, 'rotated' if rotated_attack else 'loss-based'))
    results_dir = os.path.join(ROOT_DIR,'results/{}_{}'.format(name_exp, minority_percentage))
    cifar10_experiment(
        results_dir=results_dir,
        minority_percentage=minority_percentage,
        rotated_attack=rotated_attack,
        advers=loss_based_attack,
        epsilon=epsilon
    )

