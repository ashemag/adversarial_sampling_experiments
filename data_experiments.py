from data_providers import *
from models.densenet import *
import numpy as np
import globals
import os
import torch
import csv
from torchvision import transforms
import argparse
from sklearn.model_selection import train_test_split
from experiment.base import ExperimentBuilder
import torch.optim as optim

BATCH_SIZE = 64
LEARNING_RATE = .1
WEIGHT_DECAY = 1e-4
MOMENTUM = .9


class Experiment(object):
    @staticmethod
    def _train_evaluate(model, model_title, train_data, valid_data, num_epochs, optimizer, scheduler):
        model_save_dir = os.path.join(globals.ROOT_DIR, 'SavedModels/' + model_title)
        train_results_file = os.path.join(globals.ROOT_DIR, 'ExperimentResults/' + model_title + '_train.txt')
        valid_results_file = os.path.join(globals.ROOT_DIR, 'ExperimentResults/' + model_title + '_valid.txt')

        train = (train_data, train_results_file)
        valid = (valid_data, valid_results_file)
        bpm = ExperimentBuilder(model).train_and_evaluate(num_epochs, optimizer, model_save_dir, train, valid)
        return bpm

    def _compare(self, train_data_full, train_data_full_title, train_data_mod, train_data_mod_title, test_data, num_epochs):
        # TRAIN FULL
        model_full = DenseNet(num_classes=10, depth=100, growth_rate=12, bottleneck=True, reduction=.5, dropRate=0.0)
        optimizer = torch.optim.SGD(model_full.parameters(), lr=LEARNING_RATE,
                                    momentum=MOMENTUM,
                                    nesterov=True,
                                    weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0.0001)

        bpm_full = self._train_evaluate(model_full, train_data_full_title,
                                        train_data_full, test_data, num_epochs, optimizer, scheduler)

        # TRAIN REDUCED
        model_mod = DenseNet(num_classes=10, depth=100, growth_rate=12, bottleneck=True, reduction=.5, dropRate=0.0)
        optimizer = torch.optim.SGD(model_mod.parameters(), lr=LEARNING_RATE,
                                    momentum=MOMENTUM,
                                    nesterov=True,
                                    weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0.0001)
        bpm_mod = self._train_evaluate(model_mod, train_data_mod_title,
                                       train_data_mod, test_data, num_epochs, optimizer, scheduler)

        output = {"Train_Acc_Full": bpm_full['train_acc'], "Train_Loss_Full": bpm_full['train_loss'],
                  "Train_Acc_Mod": bpm_mod['train_acc'], "Train_Loss_Mod": bpm_mod['train_loss'],
                  "Valid_Acc_Mod": bpm_mod['valid_acc'], "Valid_Loss_Mod": bpm_mod['valid_loss'],
                  "Valid_Acc_Full": bpm_full['valid_acc'], "Valid_Loss_Full": bpm_full['valid_loss'],
                  'BPM Epoch Full': bpm_full['epoch'], 'BPM Epoch Mod': bpm_mod['epoch']}
        return output


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def get_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def get_args():
    parser = argparse.ArgumentParser(description='Minority class data experiment.')
    parser.add_argument('--label')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--target_percentage', type=int)
    args = parser.parse_args()
    return args.label, args.seed, args.num_epochs, args.target_percentage


def cifar_driver():
    transform = get_transform()
    label, seed, num_epochs, target_percentage = get_args()
    title_id = 'vgg_' + label + '_' + str(target_percentage) + '_' + str(seed)
    target_percentage = target_percentage / 100

    torch.manual_seed(seed=seed)
    rng = np.random.RandomState(seed) # FOR DATA PROVIDER

    d = unpickle('data/cifar-10-batches-py/batches.meta')
    labels = d[b'label_names']
    label_mapping = {value.decode('ascii'): index for index, value in enumerate(labels)}
    print("Setting percentage reduction to {0} for label {1}".format(target_percentage, label))

    m = ModifyDataProvider()
    fieldnames = ['Target Percentage (in %)', 'Label', 'Seed', 'Num Epochs', 'Train_Loss_Full', 'Valid_Loss_Full',
                  'Valid_Loss_Mod', 'Train_Loss_Mod', 'Train_Acc_Full', 'BPM Epoch Full', 'Train_Acc_Mod', 'Valid_Acc_Full',
                  'Valid_Acc_Mod', 'BPM Epoch Mod']
    # with open('data/minority_classes_output.csv', 'a') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #
    # exit()
    train_set = CIFAR10(root='data', set_name='train', transform=transform)

    # convert inputs to numpy array instead of PIL Image
    train_inputs = np.array([np.array(i[0]) for i in train_set])
    train_targets = np.array([i[1] for i in train_set])

    train_inputs, valid_inputs, train_targets, valid_targets = train_test_split(train_inputs, train_targets,
                                                                              test_size=0.05, random_state=1)

    train_mod_inputs, train_mod_targets = m.modify(label_mapping[label], target_percentage, train_inputs, train_targets)

    m.get_label_distribution([labels[i] for i in train_targets], "full")
    m.get_label_distribution([labels[i] for i in train_mod_targets], "reduced")

    # PROCESS test data
    test_set = CIFAR10(root='data', set_name='test', transform=transform)
    # m.get_label_distribution([labels[i[1]] for i in test_set], "Test Set Full")
    test_inputs = np.array([np.array(i[0]) for i in test_set])
    test_targets = np.array([i[1] for i in test_set])

    test_set = DataProvider(test_inputs, test_targets, batch_size=BATCH_SIZE)
    valid_set = DataProvider(valid_inputs, valid_targets, batch_size=BATCH_SIZE)

    # TRAIN
    train_set_full = DataProvider(train_inputs, train_targets, batch_size=BATCH_SIZE, rng=rng)
    train_set_mod = DataProvider(train_mod_inputs, train_mod_targets, batch_size=BATCH_SIZE)

    output = Experiment()._compare(train_set_full, 'train_full_' + title_id, train_set_mod, 'train_mod_' + title_id,
                                   valid_set, num_epochs)
    output["Target Percentage (in %)"] = target_percentage * 100
    output["Label"] = label
    output["Seed"] = seed
    output["Num Epochs"] = num_epochs

    with open('data/minority_classes_output.csv', 'a') as csvfile:
        print(output)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(output)


# driver
cifar_driver()
