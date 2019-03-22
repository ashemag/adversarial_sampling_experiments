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
import torch.optim as optim
from models.base import *
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
        bpm = model.train_and_evaluate(num_epochs, optimizer, model_save_dir, train, scheduler, valid)
        return bpm

    def _compare(self, train_data, train_data_title, valid_data, num_epochs):
        model = DenseNet121()
        model = model.to(model.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                                    momentum=MOMENTUM,
                                    nesterov=True,
                                    weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0001)

        bpm = self._train_evaluate(model, train_data_title, train_data, valid_data, num_epochs, optimizer, scheduler)

        output = {"Model_Name": train_data_title, "Train_Acc": bpm['train_acc'], "Train_Loss": bpm['train_loss'],
                  "Valid_Acc": bpm['valid_acc'], "Valid_Loss": bpm['valid_loss'],
                  'BPM_Epoch': bpm['epoch']}
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
    parser.add_argument('--train_data', type=str, default='full')  # full or reduced

    args = parser.parse_args()
    return args.label, args.seed, args.num_epochs, args.target_percentage, args.train_data


def cifar_driver():
    transform = get_transform()
    label, seed, num_epochs, target_percentage, train_data_type = get_args()
    title_id = 'DN_' + label + '_' + str(target_percentage) + '_' + str(seed)
    target_percentage = target_percentage / 100

    torch.manual_seed(seed=seed)
    rng = np.random.RandomState(seed) # FOR DATA PROVIDER

    d = unpickle('data/cifar-10-batches-py/batches.meta')
    labels = d[b'label_names']
    label_mapping = {value.decode('ascii'): index for index, value in enumerate(labels)}
    print("{0}: Setting percentage reduction to {1} for label {2}".format(title_id, target_percentage, label))

    m = ModifyDataProvider()
    fieldnames = ["Model_Name", "Train_Acc", "Train_Loss",
                  "Valid_Acc", "Valid_Loss",
                  'BPM_Epoch']
    # with open('data/minority_classes_output.csv', 'w') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #
    # exit()
    train_set = CIFAR10(root='data', set_name='train', transform=transform)

    # PROCESS test data
    test_set = CIFAR10(root='data', set_name='test', transform=transform)
    test_inputs = np.array([np.array(i[0]) for i in test_set])
    test_targets = np.array([i[1] for i in test_set])
    test_set = DataProvider(test_inputs, test_targets, batch_size=BATCH_SIZE)

    # convert inputs to numpy array instead of PIL Image
    train_inputs = np.array([np.array(i[0]) for i in train_set])
    train_targets = np.array([i[1] for i in train_set])
    train_inputs, valid_inputs, train_targets, valid_targets = train_test_split(train_inputs, train_targets,
                                                                              test_size=0.05, random_state=1)
    valid_set = DataProvider(valid_inputs, valid_targets, batch_size=BATCH_SIZE)

    # TRAIN
    if train_data_type == 'full':
        train_set = DataProvider(train_inputs, train_targets, batch_size=BATCH_SIZE, rng=rng)
        m.get_label_distribution([labels[i] for i in train_targets], "full")
        title_id = 'DN_' + label + '_' + seed
    else:
        train_mod_inputs, train_mod_targets = m.modify(label_mapping[label], target_percentage, train_inputs,
                                                       train_targets)
        train_set = DataProvider(train_mod_inputs, train_mod_targets, batch_size=BATCH_SIZE)
        m.get_label_distribution([labels[i] for i in train_mod_targets], "reduced")

    output = Experiment()._compare(train_set, 'train_' + train_data_type + '_' + title_id, valid_set, num_epochs)

    with open('data/minority_classes_output.csv', 'a') as csvfile:
        print(output)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(output)


# driver
cifar_driver()
