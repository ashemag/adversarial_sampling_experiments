import time

from data_providers import *
from models.densenet import *
from globals import ROOT_DIR
from experiment_utils import unpickle
import csv
from torchvision import transforms
import argparse
import torch.optim as optim
from experiment_builder import ExperimentBuilder
from experiment_utils import set_device, get_cifar_labels_to_ints, get_cifar_ints_to_labels, get_args

BATCH_SIZE = 2048
LEARNING_RATE = .1
WEIGHT_DECAY = 1e-4
MOMENTUM = .9


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


def prepare_data(full_flag=False, batch_size=2048, minority_class=3, minority_percentage=0.01):
    """

    :param full_flag: if we are reducing class or not
    :param minority_class: minority class idx (int)
    :param minority_percentage: percentage (float)
    :return:
    """
    percentages = [1. for i in range(10)]

    # LOAD VALID DATA
    valid_set = CIFAR10(root='data', transform=get_transform('valid'), download=True, set_name='val',
                        percentages_list=percentages)
    valid_data = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=4)

    # LOAD TEST DATA
    test_set = CIFAR10(root='data', transform=get_transform('valid'), download=True, set_name='test',
                       percentages_list=percentages)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)

    # REDUCE MINORITY CLASS IN TRAINING
    minority_class_idx = -1
    if not full_flag:
        percentages[minority_class] = minority_percentage
        minority_class_idx = minority_class

    train_set = CIFAR10(root='data', transform=get_transform('train'), download=True, set_name='train',
                        percentages_list=percentages)
    train_data = MinorityDataLoader(train_set,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=4,
                                    minority_class_idx=minority_class_idx)

    return train_data, valid_data, test_data


if __name__ == "__main__":
    start_time = time.time()
    args = get_args()
    if args.full_flag:
        model_title = args.label + '_full_' + str(args.seed)
    else:
        model_title = args.label + '_' + str(args.target_percentage) + '%_' + str(args.seed)
    target_percentage = args.target_percentage / 100
    print("=== Experiment ===\n{}".format(model_title))

    device = set_device(args.seed)
    labels_to_ints = get_cifar_labels_to_ints()
    ints_to_labels = get_cifar_ints_to_labels()

    train_data, valid_data, test_data = prepare_data(full_flag=args.full_flag,
                                                     minority_class=labels_to_ints[args.label],
                                                     minority_percentage=target_percentage)

    # EXPERIMENT MODEL
    model = DenseNet121()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=LEARNING_RATE,
                                momentum=MOMENTUM,
                                nesterov=True,
                                weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0.0001)

    experiment = ExperimentBuilder(
        model=model,
        device=device,
        label_mapping=ints_to_labels,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    experiment.run_experiment(num_epochs=args.num_epochs)
    print("=== Total experiment runtime (min): {:0.2f} ===".format(round((time.time() - start_time) / float(60), 4)))
