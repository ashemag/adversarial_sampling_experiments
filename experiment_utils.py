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
    args = parser.parse_args()
    arg_str = [(str(key), str(value)) for (key, value) in vars(args).items()]
    print("=== Args ===\n {}".format(arg_str))
    return args


def prepare_output_file(filename, output=None, clean_flag=False):
    file_exists = os.path.isfile(filename)
    if clean_flag:
        if file_exists:
            os.remove(filename)
    else:
        if output is None:
            raise ValueError("Please specify output to write to output file.")
        with open(filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(output.keys()))
            if not file_exists:
                writer.writeheader()
            print("Writing to file {0}".format(filename))
            print(output)
            writer.writerow(output)
