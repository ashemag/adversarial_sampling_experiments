"""
Runs training, validation, and test experiments
"""
from comet_ml import Experiment

import time
import numpy as np
import torch
import random
import os
from tqdm import tqdm
from collections import OrderedDict
import torch.nn as nn
from collections import defaultdict
import warnings
from experiment_utils import (log_results,
                              compute_evaluation_metrics,
                              create_folder,
                              prepare_output_file,
                              remove_excess_models, get_transform, plot_confusion_matrix)

# mute warnings for sklearn
def warn(*args, **kwargs):
    pass


warnings.warn = warn


ENABLE_COMET = True
DEBUG = False


class ExperimentBuilder(nn.Module):
    def __init__(self, model, device, train_data, valid_data, test_data,
                 optimizer, scheduler, label_mapping, experiment_folder,
                 comet_experiment, attacks=None, num_classes=10):

        super(ExperimentBuilder, self).__init__()
        self.comet_experiment = comet_experiment
        self.model = model
        self.device = device
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        self.attacks = attacks
        self.attack_counter = 0

        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU
        self.scheduler = scheduler
        self.label_mapping = label_mapping

        self.best_val_model_criteria = 0.
        self.best_val_model_idx = 0.
        self.confusion_matrix = torch.zeros(num_classes, num_classes)

        # clear cached memory
        torch.cuda.empty_cache()

        # send model to device
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model).cuda()
            self.model.to(self.device)
            self.model = self.model.module
        else:
            self.model.to(self.device)  # sends the model from the cpu to the gpu

        # saving
        self.experiment_folder = experiment_folder
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))
        create_folder(self.experiment_folder)
        create_folder(self.experiment_saved_models)

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx
        and best val acc to be compared with the future val accuracies,
        in order to choose the best val model

        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        """
        path = os.path.join(model_save_dir, "{}_{}.pt".format(model_save_name, str(model_idx)))
        self.load_state_dict(torch.load(f=path))

    def save_model(self, model_save_dir, model_save_name, model_idx):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        """
        path = os.path.join(model_save_dir, "{}_{}.pt".format(model_save_name, str(model_idx)))
        torch.save(self.state_dict(), f=path)

    def save_best_performing_model(self, epoch_stats, epoch_idx):
        """
        Saves best performing model from validation f score
        :param epoch_stats: data statistics for epoch
        :param epoch_idx: epoch index to save model
        :return:
        """
        criteria = epoch_stats['valid_f_score']
        if criteria > self.best_val_model_criteria:
            self.best_val_model_criteria = criteria
            self.best_val_model_idx = epoch_idx

    def populate_iter_stats(self, loss, y, predicted, experiment_key, stats):
        """
        :param loss: loss from evaluating model
        :param y: true labels
        :param predicted: predicted labels
        :param experiment_key: train, valid, or test
        :param stats: statistics to collect for iteration
        :return:
        """
        eval_metrics = compute_evaluation_metrics(y, predicted, label_mapping=self.label_mapping)
        for key, value in eval_metrics.items():
            stats['{}_{}'.format(experiment_key, key)].append(value)

        stats[experiment_key + '_acc'].append(np.mean(list(predicted.eq(y.data).cpu())))
        stats[experiment_key + '_loss'].append(np.mean(loss.data.detach().cpu().numpy()))

    def run_train_iter(self, x, y, stats, experiment_key='train'):
        """
        Receives the inputs and targets for the model and runs a training iteration.
        Populates epoch stats, changing dict by reference
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :param experiment_key: train, to save in eval_metrics dict
        :return: the loss and accuracy for this batch
        """
        # sets model to training mode
        # (in case batch normalization or other methods have different procedures for training and evaluation)
        self.train()
        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        x = x.to(self.device)
        y = y.to(self.device)
        out = self.model.forward(x)  # forward the data in the model

        loss = self.criterion(out, y)
        loss.backward()  # backpropagate to compute gradients for current iter loss
        self.optimizer.step()  # update network parameters

        _, predicted = torch.max(out.data, 1)  # get argmax of predictions
        self.populate_iter_stats(loss, y, predicted, experiment_key, stats)

    def update_confusion_matrix(self, predicted, y):
        """
        Updates confusion matrix for final evaluation of best performing model
        :param predicted: predicts classes
        :param y: true classes
        :return:
        """
        for t, p in zip(y.data.view(-1), predicted.cpu().view(-1)):
            self.confusion_matrix[t.long(), p.long()] += 1

    def apply_attacks(self, x, y):
        if self.attack is None:
            return x
        experiment_name = self.experiment_folder.split('results/')[1]
        filename = 'images/{}_{}'.format(self.experiment_folder, experiment_name, self.attack_counter)
        self.attack_counter += 1

        # choose from self.attack at random
        attack = random.choice(self.attack)

        return attack(x, self.model, y, True, filename)

    def run_evaluation_iter(self, x, y, stats, experiment_key='valid'):
        """
        Receives the inputs and targets for the model and runs an evaluation iterations.
        Populates epoch stats, changing dict by reference
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :param experiment_key: test or valid, for eval_metrics dict
        :return: the loss and accuracy for this batch
        """
        self.eval()  # sets the system to validation mode
        x = x.to(self.device)
        y = y.to(self.device)
        out = self.model.forward(x)
        loss = self.criterion(out, y)

        _, predicted = torch.max(out.data, 1)  # get argmax of predictions
        self.populate_iter_stats(loss, y, predicted, experiment_key, stats)

        if experiment_key == 'test':
            self.update_confusion_matrix(predicted, y)
        return predicted

    def train_validation_experiments(self, num_epochs):
        train_stats = OrderedDict()
        for epoch_idx in range(num_epochs):
            epoch_start_time = time.time()
            raw_epoch_stats = defaultdict(list)
            with tqdm(total=len(self.train_data)) as pbar_train:  # create a progress bar for training
                for x, y in self.train_data:  # get data batches
                    inputs = self.apply_attack(x, y)
                    self.run_train_iter(x=inputs, y=y, stats=raw_epoch_stats)  # take a training iter step
                    pbar_train.update(1)
                    pbar_train.set_description(
                        "{} | Epoch {} | f-score {:.4f}"
                        .format('Train', epoch_idx, np.mean(raw_epoch_stats['train_f_score'])))

            with tqdm(total=len(self.valid_data)) as pbar_valid:  # create a progress bar for validation
                for x, y in self.valid_data:  # get data batches
                    inputs = self.apply_attack(x, y)
                    self.run_evaluation_iter(x=inputs, y=y, stats=raw_epoch_stats)  # run a validation iter
                    pbar_valid.update(1)  # add 1 step to the progress bar
                    pbar_valid.set_description(
                        "{} | Epoch {} | f-score {:.4f}"
                        .format('Valid', epoch_idx, np.mean(raw_epoch_stats['valid_f_score'])))

            # learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            raw_epoch_stats['learning_rate'] = self.optimizer.param_groups[0]['lr']

            # save epoch stats to training stats
            epoch_stats = {}
            for key, value in raw_epoch_stats.items():
                epoch_stats[key] = np.mean(value)
                if ENABLE_COMET:
                    self.comet_experiment.log_metric(name=key, value=epoch_stats[key], step=epoch_idx)
            epoch_stats['epoch'] = epoch_idx
            train_stats["epoch_{}".format(epoch_idx)] = epoch_stats

            if DEBUG:
                log_results(raw_epoch_stats, epoch_start_time, epoch_idx)

            self.save_model(model_save_dir=self.experiment_saved_models,
                            model_save_name="train_model",
                            model_idx=epoch_idx)
            self.save_best_performing_model(epoch_stats=epoch_stats, epoch_idx=epoch_idx)
        return train_stats

    def log_sample_test_images(self, preds, y, x):
        sample_size = 5
        indices = np.random.choice(len(preds.cpu()), sample_size)
        pred_samples = preds[indices]
        true_samples = y[indices]
        pred_samples_labels = [self.label_mapping[value.item()] for value in pred_samples]
        true_samples_labels = [self.label_mapping[value.item()] for value in true_samples]
        x_samples = x[indices]
        for i, sample in enumerate(x_samples):
            transform = get_transform(set_name='test', inverse=True)
            sample = transform(sample)
            sample = sample.permute([1, 2, 0])
            name = 'Pred {} | True {}'.format(pred_samples_labels[i], true_samples_labels[i])
            self.comet_experiment.log_image(sample, name=name)

    def log_confusion_matrix(self):
        matrix = []
        for row in self.confusion_matrix:
            norm = row.sum()
            row_vals = []
            for item in row:
                row_vals.append(item.item() / norm * 100)
            matrix.append(row_vals)
        classes = [self.label_mapping[i] for i in range(len(self.label_mapping))]
        img = plot_confusion_matrix(torch.Tensor(matrix), classes)
        self.comet_experiment.log_image(img, name='Confusion Matrix')

    def test_experiments(self):
        print("Generating test set evaluation metrics with best model index {}".format(self.best_val_model_idx))
        self.load_model(model_save_dir=self.experiment_saved_models,
                        model_idx=self.best_val_model_idx,
                        model_save_name="train_model")

        remove_excess_models(self.experiment_folder, self.best_val_model_idx)

        raw_test_stats = defaultdict(list)
        with tqdm(total=len(self.test_data)) as pbar_test:  # ini a progress bar
            for i, (x, y) in enumerate(self.test_data):  # sample batch
                inputs = self.apply_attack(x, y)
                preds = self.run_evaluation_iter(x=inputs, y=y, stats=raw_test_stats, experiment_key='test')
                pbar_test.update(1)  # update progress bar status
                pbar_test.set_description("{} | f-score {:.4f}"
                                          .format('Test', np.mean(raw_test_stats['test_f_score'])))
        # save to test stats
        test_stats = {}
        for key, value in raw_test_stats.items():
            test_stats[key] = np.mean(value)
            if ENABLE_COMET:
                self.comet_experiment.log_metric(name=key, value=test_stats[key])

        # Log confusion matrix & samples + labels
        self.log_confusion_matrix()
        self.log_sample_test_images(preds, y, x)
        return test_stats

    def aggregate_experiment_statistics(self, test_stats, train_stats, seed, experiment_name, num_epochs):
        merge_dict = dict(list(test_stats.items()) +
                          list(train_stats["epoch_{}".format(self.best_val_model_idx)].items()))

        merge_dict['epoch'] = self.best_val_model_idx
        merge_dict['seed'] = seed
        merge_dict['title'] = experiment_name
        merge_dict['num_epochs'] = num_epochs

        for key, value in merge_dict.items():
            if isinstance(value, float):
                merge_dict[key] = np.around(value, 4)

        return merge_dict

    def run_experiment(self, num_epochs, seed, experiment_name):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        train_stats = self.train_validation_experiments(num_epochs)
        prepare_output_file(
            filename="{}/{}".format(self.experiment_folder, "train_statistics_{}.csv".format(seed)),
            output=list(train_stats.values()))
        test_stats = self.test_experiments()
        stats = self.aggregate_experiment_statistics(test_stats, train_stats, seed, experiment_name, num_epochs)
        prepare_output_file(filename="{}/{}".format(self.experiment_folder, "results.csv"),
                            output=[stats])

