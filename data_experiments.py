from data_providers import *
from ModelBuilder.densenet import *
import numpy as np
import globals
import os
import torch
import torch.optim as optim
import csv
from torchvision import transforms
import argparse


class Experiment(object):
    @staticmethod
    def _train(model, model_title, train_data, num_epochs, optimizer):
        saved_models_dir = os.path.join(globals.ROOT_DIR, 'SavedModels/' + model_title)
        train_results_file = os.path.join(globals.ROOT_DIR, 'ExperimentResults/' + model_title + '.txt')
        model.train_full(train_data, num_epochs, optimizer, train_results_file, saved_models_dir)

        with open('ExperimentResults/' + model_title + '.txt') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter='\t')
            for row in csv_reader:
                train_acc, train_loss = float(row['train_acc']), float(row['train_loss'])
        return train_acc, train_loss

    @staticmethod
    def _evaluate(model, model_title, valid_data, num_epochs):
        saved_models_dir = os.path.join(globals.ROOT_DIR, 'SavedModels/' + model_title)
        eval_results_file = os.path.join(globals.ROOT_DIR, 'ExperimentResults/' + model_title + '_eval.txt')
        model.evaluate_full(valid_data, num_epochs, saved_models_dir, eval_results_file)

        with open('ExperimentResults/' + model_title + '_eval.txt') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter='\t')
            for row in csv_reader:
                print(row['train_acc'], row['train_loss'])
                eval_acc, eval_loss = float(row['train_acc']), row['train_loss']
                if eval_loss == 'None':
                    continue
        return eval_acc, float(eval_loss)

    def _compare(self, train_data_full, train_data_full_title, train_data_mod, train_data_mod_title, test_data, num_epochs):
        # TRAIN FULL
        model_full = DenseNet(nClasses=10, growthRate=32, depth=16, reduction=1, bottleneck=True)
        optimizer = optim.Adam(model_full.parameters(), amsgrad=False, weight_decay=0.00002)
        model_full = model_full.to(model_full.device)

        train_acc_full, train_loss_full = self._train(model_full, train_data_full_title, train_data_full, num_epochs, optimizer)
        valid_acc_full, valid_loss_full = self._evaluate(model_full, train_data_full_title, test_data,
                                                         [i for i in range(num_epochs)])

        # TRAIN REDUCED
        model_mod = DenseNet(nClasses=10, growthRate=32, depth=16, reduction=1, bottleneck=True)
        optimizer = optim.Adam(model_mod.parameters(), amsgrad=False, weight_decay=0.00002)
        model_mod = model_mod.to(model_mod.device)

        train_acc_mod, train_loss_mod = self._train(model_mod, train_data_mod_title, train_data_mod, num_epochs, optimizer)
        valid_acc_mod, valid_loss_mod = self._evaluate(model_mod, train_data_mod_title, test_data,
                                                       [i for i in range(num_epochs)])

        output = {"Train_Acc_Full": train_acc_full, "Train_Loss_Full": train_loss_full,
                  "Train_Acc_Mod": train_acc_mod, "Train_Loss_Mod": train_loss_mod,
                  "Valid_Acc_Mod": valid_acc_mod, "Valid_Loss_Mod": valid_loss_mod,
                  "Valid_Acc_Full": valid_acc_full, "Valid_Loss_Full": valid_loss_full}
        return output


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def get_args():
    parser = argparse.ArgumentParser(description='Minority class data experiments.')
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
                  'Valid_Loss_Mod', 'Train_Loss_Mod', 'Train_Acc_Full', 'Train_Acc_Mod', 'Valid_Acc_Full',
                  'Valid_Acc_Mod']
    # with open('data/minority_classes_output.csv', 'a') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #
    # exit()
    train_set = CIFAR10(root='data', set_name='train', transform=transform)

    # convert inputs to numpy array instead of PIL Image
    train_inputs = np.array([np.array(i[0]) for i in train_set])
    train_targets = np.array([i[1] for i in train_set])
    train_mod_inputs, train_mod_targets = m.modify(label_mapping[label], target_percentage, train_inputs, train_targets)

    m.get_label_distribution([labels[i] for i in train_targets], "full")
    m.get_label_distribution([labels[i] for i in train_mod_targets], "reduced")

    # PROCESS test data
    test_set = CIFAR10(root='data', set_name='test', transform=transform)
    m.get_label_distribution([labels[i[1]] for i in test_set], "Test Set Full")
    test_inputs = np.array([np.array(i[0]) for i in test_set])
    test_targets = np.array([i[1] for i in test_set])
    test_set = DataProvider(test_inputs, test_targets, batch_size=100)

    # TRAIN
    train_set_full = DataProvider(train_inputs, train_targets, batch_size=100, rng=rng)
    train_set_mod = DataProvider(train_mod_inputs, train_mod_targets, batch_size=100)

    output = Experiment()._compare(train_set_full, 'train_full_' + title_id, train_set_mod, 'train_mod_' + title_id,
                                   test_set, num_epochs)
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
