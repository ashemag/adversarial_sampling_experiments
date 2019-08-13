"""
Utils used by base experiment builder class
"""
import csv
import time
import os
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import numpy as np
import argparse
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image

def log_results(stats, start_time, epoch):
    """
    Log results to terminal
    :param stats: statistics to log
    :param start_time: start_time of experiments
    :param epoch: epoch index
    :return:
    """
    out_string = "".join(["{}: {:0.4f}\n".format(key, value)
                          for key, value in stats.items() if key != 'epoch'])
    epoch_elapsed_time = (time.time() - start_time) / 60  # calculate time taken for epoch
    epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
    print("\n===Epoch {}===\n{}===Elapsed time: {} mins===".format(epoch, out_string, epoch_elapsed_time))


def remove_excess_models(experiment_folder, best_val_model_idx):
    """
    Crawls directory to remove saved_models that are not best performing model
    :param experiment_folder: experiment folder to crawl
    :param best_val_model_idx: best performing model index
    :return:
    """
    dir_list_list = [dir_names for (_, dir_names, _) in os.walk(experiment_folder)]
    for dir_list in dir_list_list:
        if 'saved_models' in dir_list:
            path = os.path.abspath(os.path.join(experiment_folder, 'saved_models'))
            file_list_list = [file_names for (_, _, file_names) in os.walk(path)]
            for file_list in file_list_list:
                for file in file_list:
                    epoch = file.split('_')[-1]
                    epoch = epoch.replace('.pt', '')
                    if int(epoch) != best_val_model_idx:
                        os.remove(os.path.join(path, file))


def compute_evaluation_metrics(y_true, y_pred, label_mapping):
    """
    Computes evaluation metrics for model performances
    :param y_true: actual labels
    :param y_pred: predicts values
    :param type_key: train, valid, or test
    :param label_mapping: maps order of f_scores into labels
    :return:
    """
    stats = {}
    f1_overall = f1_score(
        y_true.cpu().detach().numpy(),
        y_pred.cpu().detach().numpy(),
        average='weighted'
    )

    stats['f_score'] = f1_overall

    precision_overall = precision_score(
        y_true.cpu().detach().numpy(),
        y_pred.cpu().detach().numpy(),
        average='weighted'
    )

    stats['precision'] = precision_overall

    recall_overall = recall_score(
        y_true.cpu().detach().numpy(),
        y_pred.cpu().detach().numpy(),
        average='weighted'
    )

    stats['recall'] = recall_overall

    f1 = f1_score(
        y_true.cpu().detach().numpy(),
        y_pred.cpu().detach().numpy(),
        average=None
    )
    precision = precision_score(
        y_true.cpu().detach().numpy(),
        y_pred.cpu().detach().numpy(),
        average=None
    )

    recall = recall_score(
        y_true.cpu().detach().numpy(),
        y_pred.cpu().detach().numpy(),
        average=None
    )

    for i in range(len(f1)):
        stats['f_score_' + label_mapping[i]] = f1[i]
        stats['precision_' + label_mapping[i]] = precision[i]
        stats['recall_' + label_mapping[i]] = recall[i]

    return stats


def unpickle(file):
    """
    Load data from pickled object
    :param file: filename to load data from
    :return:
    """
    import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def set_device(seed):
    """
    Generates device and sets randomness of experiments
    :param seed:
    :return:
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():  # checks whether a cuda gpu is available and whether the gpu flag is True
        device_local = torch.cuda.current_device()
        print("Using {} GPU(s)".format(torch.cuda.device_count()))
    else:
        print("Using CPU")
        device_local = torch.device('cpu')  # sets the device to be CPU

    return device_local


def get_cifar_labels_to_ints():
    d = unpickle('data/cifar-10-batches-py/batches.meta')
    labels = d[b'label_names']
    return {value.decode('ascii'): index for index, value in enumerate(labels)}


def get_cifar_ints_to_labels():
    d = unpickle('data/cifar-10-batches-py/batches.meta')
    labels = d[b'label_names']
    return {index: value.decode('ascii') for index, value in enumerate(labels)}


def get_args():
    parser = argparse.ArgumentParser(description='Minority class data experiment.')
    parser.add_argument('--label', default='cat')
    parser.add_argument('--seed', type=int, default=28)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--target_percentage', type=float, default=-1)
    parser.add_argument('--full_flag', type=bool, default=True)  # full or reduced
    parser.add_argument('--batch_size', type=int, default=64)  # full or reduced
    args = parser.parse_args()
    arg_str = [(str(key), str(value)) for (key, value) in vars(args).items()]
    print("=== Args ===\n {}".format(arg_str))
    return args


def create_folder(folder):
    """
    Creates folder in this folder dir
    :param folder: folder dir
    :return:
    """
    if not os.path.exists(folder):  # If experiment directory does not exist
            os.makedirs(folder)  # create the experiment directory


def prepare_output_file(filename, output=None, file_action_key='a+', aggregate=False):
    file_exists = os.path.isfile(filename)

    if output is None or output == []:
        raise ValueError("Please specify output list to write to output file.")
    with open(filename, file_action_key) as csvfile:
        fieldnames = sorted(list(output[0].keys()))  # to make sure new dictionaries in diff order work okay
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists or file_action_key == 'w' or os.path.getsize(filename) == 0:
            writer.writeheader()
        for entry in output:
            writer.writerow(entry)


def print_duration(duration):
    hours, rem = divmod(duration, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def get_transform(set_name, inverse=False):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    if inverse:
        return transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.] , std=[1/item for item in std]),
            transforms.Normalize(mean=[-item for item in mean], std=[1., 1., 1.])
        ])
    else:
        if set_name == 'train':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])


def plot_confusion_matrix(cm, classes, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
#     fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            sample = '{}%'.format(np.around(cm[i,j].item(), 1))
            ax.text(j, i, format(sample),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.grid(False)
    # plt.axis('off')
    plt.savefig('results/cm.png')
    return Image.open('results/cm.png')
