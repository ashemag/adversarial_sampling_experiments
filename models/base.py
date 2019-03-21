import numpy as np
import torch
import torch.nn.functional as F
import time
import os
from models import storage_utils
from tqdm import tqdm
import sys
from collections import OrderedDict
import torch.nn as nn
import math
from attacks.data_augmenter import DataAugmenter
from data_subsetter import DataSubsetter
from data_providers import DataProvider

TARGET = 8 # ship


class Logger(object):
    def __init__(self,disable=False,stream=sys.stdout,filename=None):
        self.disable = disable
        self.stream = stream
        self.module_name = filename

    def error(self,str):
        if not self.disable: sys.stderr.write(str+'\n')

    def print(self, obj):
        if not self.disable: self.stream.write(str(obj))
        if self.module_name: self.stream.write('. {}'.format(os.path.splitext(self.module_name)[0]))
        self.stream.write('\n')


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.num_epochs = None
        self.train_data = None
        self.optimizer = None
        self.train_file_path = None
        self.cross_entropy = None
        self.scheduler = None
        self.device = torch.device('cpu')  # sets the device to be CPU

    def use_gpu(self,gpu_ids='0'):
        logger = Logger(stream=sys.stderr)
        logger.disable = False # if disabled does not print info messages.
        logger.module_name = __file__
        gpu_available = False
        if not gpu_available: return
        if not torch.cuda.is_available(): raise Exception('system does not have any cuda device available.')

        if ',' in gpu_ids:
            self.device = [torch.device('cuda:{}'.format(idx)) for idx in gpu_ids.split(",")]
        else:
            self.device = torch.device('cuda:{}'.format(gpu_ids))

        if type(self.device) is list:
            self.device = self.device[0]

        logger.print("gpu ids being used: {}".format(gpu_ids))

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids  # (1)
        self.cuda() # (2)

        xx = next(self.parameters()).is_cuda
        logger.print("is model cuda: {}".format(xx))

        '''
        remarks:
        (1) sets the main GPU to be the one at index 0 (on multi gpu machines you can choose which one you want to use 
        by using the relevant GPU ID)
        (2) this makes model a cuda model. makes it so that gpu is used with it.
        '''

    def get_acc_batch(self,x_batch,y_batch, y_batch_pred=None, integer_encoded=False):
        """
        :param x_batch: array or tensor
        :param y_batch: array, one-hot-encoded
        :param y_batch_pred:  tensor, (because results from model forward pass)
        :return:
        """

        if type(x_batch) is np.ndarray:
            x_batch = torch.Tensor(x_batch).float().to(device=self.device)

        if y_batch_pred is None:
            y_batch_pred = self(x_batch)

        if integer_encoded:
            y_batch_int = np.int64(y_batch.reshape(-1,))
        else:
            y_batch_int = np.argmax(y_batch,axis=1)

        y_batch_int = torch.Tensor(y_batch_int).long().to(device=self.device)
        _, y_pred_batch_int = torch.max(y_batch_pred.data, 1)  # argmax of predictions
        acc = np.mean(list(y_pred_batch_int.eq(y_batch_int.data).cpu()))  # compute accuracy

        y_pred_batch_int = np.array(y_pred_batch_int.cpu())
        y_batch_int = np.array(y_batch_int.cpu())

        # get indices where truth value is not target
        indices = np.array([i if elem != TARGET else -1 for i, elem in enumerate(y_batch_int)])
        # print("\nTotal true target labels in this batch: {0}".format(len(y_batch_int) - len(indices)))
        # remove these indices from preds we are considering. length is number of truth values with target
        y_pred_batch_int_target = np.delete(y_pred_batch_int, indices)
        target_acc = np.mean([1 if y_pred_batch_int_target[i] == TARGET else 0 for i in range(len(y_pred_batch_int_target))])
        if math.isnan(target_acc):
            target_acc = 0

        return acc, target_acc

    @staticmethod
    def save_train_epoch_results(batch_statistics,train_file_path):
        statistics_to_save = {"train_acc":0, "train_loss":0, "epoch_train_time":0,"current_epoch":0}
        statistics_to_save["current_epoch"] = batch_statistics["current_epoch"]
        statistics_to_save["epoch_train_time"] = batch_statistics["epoch_train_time"]

        for key, value in batch_statistics.items():
            if key not in ["current_epoch","epoch_train_time"]:
                batch_values = np.array(batch_statistics[key])
                epoch_val = np.mean(batch_values)  # get mean of all metrics of current epoch metrics dict
                statistics_to_save[key] = np.around(epoch_val, decimals=4)

        print(statistics_to_save)
        storage_utils.save_statistics(statistics_to_save,train_file_path)

    def advers_train_and_evaluate(self,
                                  max_num_batches_minority,
                                  max_num_batches_majority,
                                  labels_minority,
                                  attack,
                                  advs_images_file,
                                  m_batch_size,
                                  o_batch_size,
                                  num_epochs,optimizer,
                                  model_save_dir,
                                  train,
                                  scheduler=None,
                                  valid=None,
                                  disable_progress=False):
        '''
        :param labels_minority: list of integers.
        :param attack:
        :param num_epochs:
        :param optimizer:
        :param model_save_dir:
        :param train:
        :param scheduler:
        :param valid:
        :return:
        '''

        logger = Logger(stream = sys.stderr,disable= False)
        logger.print('starting adversarial training procedure')
        logger.print('attack used: {}'.format(type(attack)))

        self.num_epochs = num_epochs
        self.train_data = train[0]
        self.optimizer = optimizer
        self.train_file_path = train[1]
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        if scheduler is not None:
            self.scheduler = scheduler

        x = train[0].inputs
        y = train[0].targets

        labels_majority = list(set(y)-set(labels_minority))
        xm, ym = DataSubsetter.condition_on_label(x, y, labels=labels_minority, shuffle=False, rng=None)
        xo, yo = DataSubsetter.condition_on_label(x, y, labels=labels_majority, shuffle=False, rng=None)

        minority_percentage = 0.01
        size_minority = int(len(xm) * minority_percentage)
        xm = xm[:size_minority]
        ym = ym[:size_minority]

        logger.print('minority data size: {}. majority data size: {}. total: {}'.format(len(xm),len(xo),len(xo)+len(xm)))

        dp_o = DataProvider(xo,yo,batch_size=o_batch_size,max_num_batches=max_num_batches_majority,make_one_hot=False,rng=None,with_replacement=False)
        dp_m = DataProvider(xm,ym,batch_size=m_batch_size,max_num_batches=max_num_batches_minority,make_one_hot=False,rng=None,with_replacement=True)

        def advers_train_epoch(dp_o,dp_m):
            '''
            :param dp_o:
                type: data provider.
            :param dp_m:
                type: data provider.
            :return:
            '''

            batch_statistics = {'loss': [], 'acc': [], 'target_acc':[]}
            xm_batch_adv = None

            for i, (xo_batch, yo_batch) in tqdm(enumerate(dp_o), file=sys.stderr):  # get data batches

                xm_batch, ym_batch = dp_m.__next__()
                xm_batch_adv = attack(xm_batch,ym_batch)  # DataAugmenter.advers_attack(xm_batch, ym_batch, attack=attack, disable_progress=disable_progress)
                xm_batch_comb = np.vstack((xo_batch,xm_batch,xm_batch_adv))
                ym_batch_comb = np.hstack((yo_batch,ym_batch,ym_batch))

                loss_batch, accuracy_batch, target_acc_batch = self.train_iter(xm_batch_comb, ym_batch_comb)  # process batch
                batch_statistics['loss'].append(loss_batch.item())
                batch_statistics['acc'].append(accuracy_batch)
                batch_statistics['target_acc'].append(target_acc_batch)

            epoch_loss = np.mean(np.array(batch_statistics['loss']))
            epoch_acc = np.mean(np.array(batch_statistics['acc']))
            epoch_target_acc = np.mean(np.array(batch_statistics['target_acc']))

            attack.model = self
            # print("epoch ended. updated model of attack. ")

            return epoch_loss, epoch_acc, epoch_target_acc, xm_batch_adv

        def validation_epoch(data):
            '''
            :param func: the function that determines how batches are processed. either self.train_iter, or
            self.evaluation_iter. if func==self.train_iter data is being processed for the purpose of updating the
            weights of the network.
            :param data: DataProvider object.
            :return epoch accuracy and loss.
            '''
            batch_statistics = {'loss': [], 'acc': [], 'target_acc':[]}
            for i, (x_train_batch, y_train_batch) in tqdm(enumerate(data), file=sys.stderr):  # get data batches
                loss_batch, accuracy_batch, target_accuracy_batch = self.run_evaluation_iter(x_train_batch, y_train_batch, integer_encoded=True) # process batch
                batch_statistics['loss'].append(loss_batch.item())
                batch_statistics['acc'].append(accuracy_batch)
                batch_statistics['target_acc'].append(target_accuracy_batch)

            epoch_loss = np.mean(np.array(batch_statistics['loss']))
            epoch_acc = np.mean(np.array(batch_statistics['acc']))
            epoch_target_acc = np.mean(np.array(batch_statistics['target_acc']))

            return epoch_loss, epoch_acc, epoch_target_acc

        advs_images_dict = {}

        bpm = {'valid_acc': 0}
        torch.cuda.empty_cache()
        for current_epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            train_epoch_loss, train_epoch_acc, train_epoch_target_acc, xm_batch_adv = advers_train_epoch(dp_o, dp_m)

            advs_images_dict[current_epoch] = xm_batch_adv # save batch of images as results.

            epoch_train_time = time.time() - epoch_start_time
            train_statistics_to_save = OrderedDict({
                'current_epoch': current_epoch,
                'train_acc': np.around(train_epoch_acc, decimals=4),  # round results to 4 decimals.
                'train_loss': np.around(train_epoch_loss, decimals=4),
                'train_target_acc': np.around(train_epoch_target_acc, decimals=4),
                'epoch_train_time': epoch_train_time,
            })

            import pickle
            with open(advs_images_file, 'wb') as f:
                pickle.dump(advs_images_dict, f)

            logger.print('finished training epoch {}'.format(current_epoch))
            logger.print('finished saving adversarial images epoch {}'.format(current_epoch))

            storage_utils.save_statistics(train_statistics_to_save, file_path=train[1])
            self.save_model(model_save_dir, model_save_name='model_epoch_{}'.format(str(current_epoch)))
            results_to_print = train_statistics_to_save

            if valid is not None:  # valid is a tuple. valid[0] contains the DataProvider
                valid_epoch_loss, valid_epoch_acc, valid_epoch_target_acc = validation_epoch(data=valid[0])

                valid_statistics_to_save = OrderedDict({
                    'current_epoch': current_epoch,
                    'valid_acc': np.around(valid_epoch_acc, decimals=4),
                    'valid_target_acc': np.around(valid_epoch_target_acc, decimals=4),
                    'valid_loss': np.around(valid_epoch_loss, decimals=4)
                })

                if valid_epoch_acc > bpm['valid_acc']:
                    bpm['valid_acc'] = valid_epoch_acc
                    bpm['train_acc'] = train_epoch_acc
                    bpm['epoch'] = current_epoch
                    bpm['train_loss'] = train_epoch_loss
                    bpm['valid_loss'] = valid_epoch_loss
                    bpm['valid_target_acc'] = valid_epoch_target_acc
                    bpm['train_target_acc'] = train_epoch_target_acc


                storage_utils.save_statistics(valid_statistics_to_save, file_path=valid[1])

                results_to_print = {
                    'epoch': current_epoch,
                    'best_valid_acc': bpm['valid_acc'],
                    'valid_acc': valid_epoch_acc,
                    'train_acc': train_epoch_acc,
                    'valid_loss': valid_epoch_loss,
                    'valid_target_acc': valid_epoch_target_acc,
                    'train_target_acc': train_epoch_target_acc,
                    'train_loss': train_epoch_loss,
                    'time': epoch_train_time,
                    'best_epoch': bpm['epoch']
                }

                logger.print('finished validating epoch {}'.format(current_epoch))

                if scheduler is not None: scheduler.step()

                for param_group in self.optimizer.param_groups:
                    logger.print('learning rate: {}'.format(param_group['lr']))

            logger.print(results_to_print)

        return bpm

    def train_and_evaluate(self, target, num_epochs, optimizer, model_save_dir, train, scheduler=None, valid=None):
        '''
        :param train: is a tuple (train_data, train_save_path), where train_data is a DataProvider object of the
        training-set, and train_save_path is a string that points to the file where you want to store training results
        in.
        :param valid: is a tuple (valid_data, valid_save_path), where valid_data is a DataProvider object of the
        validation-set, and valid_save_path is a string that points to the file where you want to store validation
        results in.
        :param model_save_dir: this is the directory that models during every epoch of training are saved to.

        functions:
        (1) keeps track of model that performs best on validation-set.
        (2) prints and saves training and validation results every epoch.
        (3) saves models at every epoch.
        '''

        self.num_epochs = num_epochs
        self.train_data = train[0]
        self.optimizer = optimizer
        self.train_file_path = train[1]
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        if scheduler is not None:
            self.scheduler = scheduler

        def process_data(func, data):
            '''
            :param func: the function that determines how batches are processed. either self.train_iter, or
            self.evaluation_iter. if func==self.train_iter data is being processed for the purpose of updating the
            weights of the network.
            :param data: DataProvider object.
            :return epoch accuracy and loss.
            '''
            batch_statistics = {'loss': [], 'acc': [], 'target_acc': []}

            for i, (x_train_batch, y_train_batch) in tqdm(enumerate(data), file=sys.stdout):  # get data batches
                loss_batch, accuracy_batch, target_accuracy_batch = func(x_train_batch, y_train_batch, target)  # process batch
                batch_statistics['loss'].append(loss_batch.item())
                batch_statistics['acc'].append(accuracy_batch)
                batch_statistics['target_acc'].append(target_accuracy_batch)

            epoch_loss = np.mean(np.array(batch_statistics['loss']))
            epoch_acc = np.mean(np.array(batch_statistics['acc']))
            epoch_target_acc = np.mean(np.array(batch_statistics['target_acc']))
            return epoch_loss, epoch_acc, epoch_target_acc

        bpm = {'valid_acc': 0}
        torch.cuda.empty_cache()
        for current_epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            train_epoch_loss, train_epoch_acc, train_epoch_target_acc = process_data(func=self.train_iter, data=train[0])
            epoch_train_time = time.time() - epoch_start_time
            train_statistics_to_save = OrderedDict({
                'current_epoch': current_epoch,
                'train_acc': np.around(train_epoch_acc, decimals=4),  # round results to 4 decimals.
                'train_loss': np.around(train_epoch_loss, decimals=4),
                'train_target_acc': np.around(train_epoch_target_acc, decimals=4),
                'epoch_train_time': epoch_train_time
            })
            storage_utils.save_statistics(train_statistics_to_save, file_path=train[1])
            self.save_model(model_save_dir, model_save_name='model_epoch_{}'.format(str(current_epoch)))
            results_to_print = train_statistics_to_save

            if valid is not None:  # valid is a tuple. valid[0] contains the DataProvider
                valid_epoch_loss, valid_epoch_acc, valid_epoch_target_acc = process_data(func=self.run_evaluation_iter, data=valid[0])

                valid_statistics_to_save = OrderedDict({
                    'current_epoch': current_epoch,
                    'valid_acc': np.around(valid_epoch_acc, decimals=4),
                    'valid_loss': np.around(valid_epoch_loss, decimals=4),
                    'valid_target_acc': np.around(valid_epoch_target_acc, decimals=4)
                })

                if valid_epoch_acc > bpm['valid_acc']:
                    bpm['valid_acc'] = valid_epoch_acc
                    bpm['train_acc'] = train_epoch_acc
                    bpm['epoch'] = current_epoch
                    bpm['train_loss'] = train_epoch_loss
                    bpm['valid_loss'] = valid_epoch_loss
                    bpm['valid_target_acc'] = valid_epoch_target_acc
                    bpm['train_target_acc'] = train_epoch_target_acc

                storage_utils.save_statistics(valid_statistics_to_save, file_path=valid[1])

                results_to_print = {
                    'epoch': current_epoch,
                    'best_valid_acc': bpm['valid_acc'],
                    'valid_acc': valid_epoch_acc,
                    'train_acc': train_epoch_acc,
                    'valid_loss': valid_epoch_loss,
                    'train_loss': train_epoch_loss,
                    'time': epoch_train_time,
                    'best_epoch': bpm['epoch'],
                    'train_target_acc': train_epoch_target_acc,
                    'valid_target_acc': valid_epoch_target_acc,
                }
                scheduler.step()
                for param_group in self.optimizer.param_groups:
                    sys.stderr.write("Learning rate {0}".format(param_group['lr']))

            print(results_to_print)
        return bpm

    def train_iter(self, x_train_batch, y_train_batch, integer_encoded=True):
        """
        :param x_train_batch: array
        :param y_train_batch: array, one-hot-encoded
        :return:
        """

        # CrossEntropyLoss. Input: (N,C), target: (N) each value is integer encoded.

        self.train()
        criterion = nn.CrossEntropyLoss().cuda()

        if integer_encoded:
            y_train_batch_int = np.int64(y_train_batch.reshape(-1,))
        else:
            y_train_batch_int = np.argmax(y_train_batch, axis=1)

        y_train_batch_int = torch.Tensor(y_train_batch_int).long().to(device=self.device)
        x_train_batch = torch.Tensor(x_train_batch).float().to(device=self.device)
        y_pred_batch = self.forward(x_train_batch) # model forward pass

        print("forward prop finished.")

        loss = criterion(input=y_pred_batch, target=y_train_batch_int) # self.cross_entropy(input=y_pred_batch,target=y_train_batch_int)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        acc_batch, target_acc_batch = self.get_acc_batch(x_train_batch, y_train_batch, y_pred_batch, integer_encoded=integer_encoded)
        return loss.data, acc_batch, target_acc_batch

    def run_evaluation_iter(self,x_batch,y_batch, integer_encoded=False):
        '''
        :param x_batch:
        :param y_batch:
        :return:
        '''
        with torch.no_grad():
            self.eval()
            if not integer_encoded:
                y_batch_int = np.argmax(y_batch, axis=1)
            else:
                y_batch_int = y_batch

            y_batch_int_tens = torch.Tensor(y_batch_int).long().to(device=self.device)
            x_batch_tens = torch.Tensor(x_batch).float().to(device=self.device)
            y_batch_pred_tens = self(x_batch_tens)  # model forward pass
            loss_batch = F.cross_entropy(input=y_batch_pred_tens,target=y_batch_int_tens)
            acc_batch, target_acc_batch = self.get_acc_batch(x_batch_tens,y_batch, y_batch_pred_tens)

        return loss_batch.data, acc_batch, target_acc_batch

    def save_model(self, model_save_dir,model_save_name):
        state = dict()
        state['network'] = self.state_dict()  # save network parameter and other variables.
        model_path = os.path.join(model_save_dir,model_save_name)

        directory = os.path.dirname(model_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(state, f=model_path)

    def load_model(self, model_path):
        state = torch.load(f=model_path)
        self.load_state_dict(state_dict=state['network'])