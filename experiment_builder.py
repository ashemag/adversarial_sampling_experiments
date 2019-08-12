"""
Runs training, validation, and test experiments
"""
import time
import numpy as np
import torch
import os
from tqdm import tqdm
import sys
from collections import OrderedDict
import torch.nn as nn
from collections import defaultdict
from experiment_utils import log_results, compute_evaluation_metrics
import warnings


# mute warnings for sklearn
def warn(*args, **kwargs):
    pass


warnings.warn = warn


ENABLE_COMET = False
DEBUG = False


class Logger(object):
    def __init__(self, disable=False, stream=sys.stdout, filename=None):
        self.disable = disable
        self.stream = stream
        self.module_name = filename

    def error(self, str):
        if not self.disable:
            sys.stderr.write(str + '\n')

    def print(self, obj):
        if not self.disable:
            self.stream.write(str(obj))
        if self.module_name:
            self.stream.write('. {}'.format(os.path.splitext(self.module_name)[0]))
        self.stream.write('\n')


class ExperimentBuilder(nn.Module):
    def __init__(self, model, device, train_data, valid_data, test_data,
                 optimizer, scheduler, label_mapping, num_classes=10):

        super(ExperimentBuilder, self).__init__()

        self.model = model
        self.device = device
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU
        self.scheduler = scheduler
        self.label_mapping = label_mapping

        self.best_val_model_criteria = None
        self.best_val_model_idx = None
        self.confusion_matrix = torch.zeros(num_classes, num_classes)

        # set up logger
        logger = Logger(stream=sys.stderr)
        logger.disable = False  # if disabled does not print info messages.
        logger.module_name = __file__

        # send model to device
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model).cuda()
            self.model.to(self.device)
            self.model = self.model.module
        else:
            self.model.to(self.device)  # sends the model from the cpu to the gpu

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

    # def apply_attacks(batch, advs_images_dict, current_epoch):
    #     (x_maj_batch, y_maj_batch, x_min_batch, y_min_batch) = batch
    #     x_min_batch = x_min_batch.float().to(device=self.device)
    #     y_min_batch = y_min_batch.long().to(device=self.device)
    #
    #     x_min_batch_adv = attack(x_min_batch, y_min_batch)
    #
    #     # combine
    #     x_comb_batch = torch.cat([x_maj_batch.detach().clone().cpu(),
    #                               x_min_batch.detach().clone().cpu(),
    #                               x_min_batch_adv.detach().clone().cpu()], dim=0)
    #     y_comb_batch = torch.cat([y_maj_batch.detach().clone().cpu(),
    #                               y_min_batch.detach().clone().cpu(),
    #                               y_min_batch.detach().clone().cpu()], dim=0)
    #
    #     advs_images_dict[current_epoch] = x_min_batch_adv.detach().clone().cpu().numpy()
    #
    #     return x_comb_batch, y_comb_batch

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

    def run_experiment(self, num_epochs):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        train_stats = OrderedDict()
        for epoch_idx in range(num_epochs):
            epoch_start_time = time.time()
            raw_epoch_stats = defaultdict(list)
            with tqdm(total=len(self.train_data)) as pbar_train:  # create a progress bar for training
                for x, y in self.train_data:  # get data batches
                    self.run_train_iter(x=x, y=y, stats=raw_epoch_stats)  # take a training iter step
                    pbar_train.update(1)
                    pbar_train.set_description(
                        "{} Epoch {}: f-score: {:.4f}"
                            .format('Train', epoch_idx, np.mean(raw_epoch_stats['train_f_score'])))

            with tqdm(total=len(self.valid_data)) as pbar_val:  # create a progress bar for validation
                for x, y in self.valid_data:  # get data batches
                    self.run_evaluation_iter(x=x, y=y, stats=raw_epoch_stats)  # run a validation iter
                    pbar_val.update(1)  # add 1 step to the progress bar
                    pbar_train.set_description(
                        "{} Epoch {}: f-score: {:.4f}"
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
                    self.experiment.log_metric(name=key, value=raw_epoch_stats[key], step=epoch_idx)
            epoch_stats['epoch'] = epoch_idx
            train_stats["epoch_{}".format(epoch_idx)] = raw_epoch_stats

            if DEBUG:
                log_results(raw_epoch_stats, epoch_start_time, epoch_idx)

            self.save_model(model_save_dir=self.experiment_saved_models,
                            model_save_name="train_model",
                            model_idx=epoch_idx)
            self.save_best_performing_model(epoch_stats=raw_epoch_stats, epoch_idx=epoch_idx)
