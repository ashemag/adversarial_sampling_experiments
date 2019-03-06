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

from attacks.data_augmenter import DataAugmenter
from data_subsetter import DataSubsetter
from data_providers import DataProvider

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.num_epochs = None
        self.train_data = None
        self.optimizer = None
        self.train_file_path = None
        self.cross_entropy = None
        self.scheduler = None
        use_gpu = True
        gpu_id = "1,2,3,4"

        sys.stderr.write("cuda available {}\n".format(torch.cuda.is_available()))

        if torch.cuda.is_available() and use_gpu:  # checks whether a cuda gpu is available and whether the gpu flag is True
            if "," in gpu_id:
                self.device = [torch.device('cuda:{}'.format(idx)) for idx in gpu_id.split(",")]  # sets device to be cuda
            else:
                self.device = torch.device('cuda:{}'.format(gpu_id))  # sets device to be cuda

            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  # sets the main GPU to be the one at index 0 (on multi gpu machines you can choose which one you want to use by using the relevant GPU ID)
            sys.stderr.write("use GPU; GPU ID {}\n".format(gpu_id))
        else:
            sys.stderr.write("use CPU\n")
            self.device = torch.device('cpu')  # sets the device to be CPU
        if type(self.device) is list:
            self.device = self.device[0]

    def get_acc_batch(self,x_batch,y_batch,y_batch_pred=None,integer_encoded=False):
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

        return acc

    # def train_iter(self,x_train_batch,y_train_batch):
    #     """
    #     :param x_train_batch: array, one-hot-encoded
    #     :param y_train_batch: array, one-hot-encoded
    #     :return:
    #     """
    #
    #     # CrossEntropyLoss. Input: (N,C), target: (N) each value is integer encoded.
    #
    #     self.train()
    #     y_train_batch_int = np.argmax(y_train_batch,axis=1)
    #     y_train_batch_int = torch.Tensor(y_train_batch_int).long().to(device=self.device)
    #     x_train_batch = torch.Tensor(x_train_batch).float().to(device=self.device)
    #     y_pred_batch = self(x_train_batch) # model forward pass
    #     loss = F.cross_entropy(input=y_pred_batch,target=y_train_batch_int) # self.cross_entropy(input=y_pred_batch,target=y_train_batch_int)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     acc_batch = self.get_acc_batch(x_train_batch,y_train_batch,y_pred_batch)
    #
    #     return loss.data, acc_batch

    def save_train_epoch_results(self, batch_statistics,train_file_path):
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

        print("starting adversarial training procedure")
        print("attack used: ",type(attack))

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

        sys.stderr.write("seperated minority classes training data from the other classes.\n")
        print("data points minority classes: ",len(xm), "majority classes: ",len(xo), "total: ",len(xo)+len(xm))

        dp_o = DataProvider(xo,yo,batch_size=o_batch_size,max_num_batches=2,make_one_hot=False,rng=None,with_replacement=False)
        dp_m =DataProvider(xm,ym,batch_size=m_batch_size,max_num_batches=-1,make_one_hot=False,rng=None,with_replacement=True)

        def advers_train_epoch(dp_o,dp_m):
            '''
            :param dp_o:
                type: data provider.
            :param dp_m:
                type: data provider.
            :return:
            '''

            batch_statistics = {'loss': [], 'acc': []}
            xm_batch_adv = None

            for i, (xo_batch, yo_batch) in tqdm(enumerate(dp_o), file=sys.stderr):  # get data batches
                xm_batch, ym_batch = dp_m.__next__()
                # next create advers batch. then merge everything together and do training iter as usual.
                xm_batch_adv = DataAugmenter.advers_attack(xm_batch, ym_batch, attack=attack,
                                                           disable_progress=disable_progress)
                xm_batch_comb = np.vstack((xo_batch,xm_batch,xm_batch_adv))
                ym_batch_comb = np.hstack((yo_batch,ym_batch,ym_batch))

                # xm_batch_comb = xo_batch
                # ym_batch_comb = yo_batch

                sys.stderr.write("starting training iter.\n")

                loss_batch, accuracy_batch = self.train_iter(xm_batch_comb, ym_batch_comb)  # process batch
                batch_statistics['loss'].append(loss_batch.item())
                batch_statistics['acc'].append(accuracy_batch)


            '''
            results to save:
            after each epoch save xm_batch_adv (shouldn't be too many images like 10-20 anyway per epoch).
            for now keep training acc/ loss as it is - might need to split to calc advers loss.
            main thing that is important is to see the effect on valid set.
            '''

            epoch_loss = np.mean(np.array(batch_statistics['loss']))
            epoch_acc = np.mean(np.array(batch_statistics['acc']))

            attack.model = self
            print("epoch ended. updated model of attack. ")

            return epoch_loss, epoch_acc, xm_batch_adv

        def validation_epoch(data):
            '''
            :param func: the function that determines how batches are processed. either self.train_iter, or
            self.evaluation_iter. if func==self.train_iter data is being processed for the purpose of updating the
            weights of the network.
            :param data: DataProvider object.
            :return epoch accuracy and loss.
            '''
            batch_statistics = {'loss': [], 'acc': []}

            for i, (x_train_batch, y_train_batch) in tqdm(enumerate(data), file=sys.stdout):  # get data batches
                loss_batch, accuracy_batch = self.run_evaluation_iter(x_train_batch, y_train_batch,integer_encoded=True) # process batch
                batch_statistics['loss'].append(loss_batch.item())
                batch_statistics['acc'].append(accuracy_batch)

            epoch_loss = np.mean(np.array(batch_statistics['loss']))
            epoch_acc = np.mean(np.array(batch_statistics['acc']))

            return epoch_loss, epoch_acc

        advs_images_dict = {}

        bpm = {'valid_acc': 0}
        torch.cuda.empty_cache()
        for current_epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            train_epoch_loss, train_epoch_acc, xm_batch_adv = advers_train_epoch(dp_o,dp_m)

            advs_images_dict[current_epoch] = xm_batch_adv # save batch of images as results.

            epoch_train_time = time.time() - epoch_start_time
            train_statistics_to_save = OrderedDict({
                'current_epoch': current_epoch,
                'train_acc': np.around(train_epoch_acc, decimals=4),  # round results to 4 decimals.
                'train_loss': np.around(train_epoch_loss, decimals=4),
                'epoch_train_time': epoch_train_time
            })

            import pickle
            with open(advs_images_file, 'wb') as f:
                pickle.dump(advs_images_dict, f)

            sys.stderr.write("{} finished saving advs images\n".format(current_epoch))

            storage_utils.save_statistics(train_statistics_to_save, file_path=train[1])
            self.save_model(model_save_dir, model_save_name='model_epoch_{}'.format(str(current_epoch)))
            results_to_print = train_statistics_to_save

            if valid is not None:  # valid is a tuple. valid[0] contains the DataProvider
                valid_epoch_loss, valid_epoch_acc = validation_epoch(data=valid[0])

                valid_statistics_to_save = OrderedDict({
                    'current_epoch': current_epoch,
                    'valid_acc': np.around(valid_epoch_acc, decimals=4),
                    'valid_loss': np.around(valid_epoch_loss, decimals=4)
                })

                if valid_epoch_acc > bpm['valid_acc']:
                    bpm['valid_acc'] = valid_epoch_acc
                    bpm['train_acc'] = train_epoch_acc
                    bpm['epoch'] = current_epoch
                    bpm['train_loss'] = train_epoch_loss
                    bpm['valid_loss'] = valid_epoch_loss

                storage_utils.save_statistics(valid_statistics_to_save, file_path=valid[1])

                results_to_print = {
                    'epoch': current_epoch,
                    'best_valid_acc': bpm['valid_acc'],
                    'valid_acc': valid_epoch_acc,
                    'train_acc': train_epoch_acc,
                    'valid_loss': valid_epoch_loss,
                    'train_loss': train_epoch_loss,
                    'time': epoch_train_time,
                    'best_epoch': bpm['epoch']
                }

                if scheduler is not None: scheduler.step()

                for param_group in self.optimizer.param_groups:
                    print("Learning rate ", param_group['lr'])

            print(results_to_print)

        return bpm

    def train_and_evaluate(self, num_epochs, optimizer, model_save_dir, train, scheduler = None, valid=None):
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
            batch_statistics = {'loss': [], 'acc': []}

            for i, (x_train_batch, y_train_batch) in tqdm(enumerate(data), file=sys.stdout):  # get data batches
                loss_batch, accuracy_batch = func(x_train_batch, y_train_batch)  # process batch
                batch_statistics['loss'].append(loss_batch.item())
                batch_statistics['acc'].append(accuracy_batch)

            epoch_loss = np.mean(np.array(batch_statistics['loss']))
            epoch_acc = np.mean(np.array(batch_statistics['acc']))

            return epoch_loss, epoch_acc

        bpm = {'valid_acc': 0}
        torch.cuda.empty_cache()
        for current_epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            train_epoch_loss, train_epoch_acc = process_data(func=self.train_iter, data=train[0])
            epoch_train_time = time.time() - epoch_start_time
            train_statistics_to_save = OrderedDict({
                'current_epoch': current_epoch,
                'train_acc': np.around(train_epoch_acc, decimals=4),  # round results to 4 decimals.
                'train_loss': np.around(train_epoch_loss, decimals=4),
                'epoch_train_time': epoch_train_time
            })
            storage_utils.save_statistics(train_statistics_to_save, file_path=train[1])
            self.save_model(model_save_dir, model_save_name='model_epoch_{}'.format(str(current_epoch)))
            results_to_print = train_statistics_to_save

            if valid is not None:  # valid is a tuple. valid[0] contains the DataProvider
                valid_epoch_loss, valid_epoch_acc = process_data(func=self.run_evaluation_iter, data=valid[0])

                valid_statistics_to_save = OrderedDict({
                    'current_epoch': current_epoch,
                    'valid_acc': np.around(valid_epoch_acc, decimals=4),
                    'valid_loss': np.around(valid_epoch_loss, decimals=4)
                })

                if valid_epoch_acc > bpm['valid_acc']:
                    bpm['valid_acc'] = valid_epoch_acc
                    bpm['train_acc'] = train_epoch_acc
                    bpm['epoch'] = current_epoch
                    bpm['train_loss'] = train_epoch_loss
                    bpm['valid_loss'] = valid_epoch_loss

                storage_utils.save_statistics(valid_statistics_to_save, file_path=valid[1])

                results_to_print = {
                    'epoch': current_epoch,
                    'best_valid_acc': bpm['valid_acc'],
                    'valid_acc': valid_epoch_acc,
                    'train_acc': train_epoch_acc,
                    'valid_loss': valid_epoch_loss,
                    'train_loss': train_epoch_loss,
                    'time': epoch_train_time,
                    'best_epoch': bpm['epoch']
                }
                scheduler.step()
                for param_group in self.optimizer.param_groups:
                    print("Learning rate ", param_group['lr'])

            print(results_to_print)
        return bpm

    def train_full(self, train_data, num_epochs, optimizer,train_file_path,model_save_dir,integer_encoded=False):
        self.num_epochs = num_epochs
        self.train_data = train_data
        self.optimizer = optimizer
        self.train_file_path = train_file_path
        self.cross_entropy = torch.nn.CrossEntropyLoss()

        for current_epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            batch_statistics = {"train_acc": [], "train_loss": []}

            for i, (x_train_batch, y_train_batch) in tqdm(enumerate(self.train_data), file=sys.stdout):  # get data batches
                loss_batch, accuracy_batch = self.train_iter(x_train_batch, y_train_batch,integer_encoded)  # take a training iter step
                batch_statistics["train_loss"].append(loss_batch.item())  # add current iter loss to the train loss list
                batch_statistics["train_acc"].append(accuracy_batch)  # add current iter acc to the train acc list

            statistics_to_save = \
                OrderedDict({"current_epoch": current_epoch, "train_acc": 0, "train_loss": 0, "epoch_train_time": 0})
            statistics_to_save["epoch_train_time"] = time.time() - epoch_start_time

            for key, value in batch_statistics.items():
                if key not in ["current_epoch", "epoch_train_time"]:
                    batch_values = np.array(batch_statistics[key])
                    epoch_val = np.mean(batch_values)  # get mean of all metrics of current epoch metrics dict
                    statistics_to_save[key] = np.around(epoch_val, decimals=4)

            print(statistics_to_save)
            storage_utils.save_statistics(statistics_to_save, train_file_path)
            self.save_model(model_save_dir,model_save_name='model_epoch_{}'.format(str(current_epoch))) # (1)

            '''
            Remarks:
            (1) models that are saved at each epoch are specifically given the name "model_epoch_{}". important for
            it to be in format. this format is assumed in other functions e.g. run_evaluation_iter()
            '''


    def train_iter(self, x_train_batch, y_train_batch, integer_encoded=True):
        """
        :param x_train_batch: array
        :param y_train_batch: array, one-hot-encoded
        :return:
        """

        print("entered train iter")

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

        loss = criterion(input=y_pred_batch,target=y_train_batch_int) # self.cross_entropy(input=y_pred_batch,target=y_train_batch_int)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print("finished bprop")

        acc_batch = self.get_acc_batch(x_train_batch,y_train_batch,y_pred_batch,integer_encoded=integer_encoded)

        return loss.data, acc_batch

    def run_evaluation_iter(self,x_batch,y_batch,integer_encoded=False):
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
            acc_batch = self.get_acc_batch(x_batch_tens,y_batch,y_batch_pred_tens,integer_encoded=integer_encoded)

        return loss_batch.data, acc_batch # TODO: what is the return type?

    def evaluate_full(self,valid_set,epochs,model_train_dir,eval_results_file_path):
        '''
        :param valid_set:
        :param model_train_dir:
        :param epochs:

        during training model at each epoch is saved. this method loads models at specified epochs and
        evaluates performance on a given (validation) set.

        design: seperate training from testing on validation set. sometimes more convenient.
        '''
        for epoch in epochs:
            load_from_path = os.path.join(model_train_dir,'model_epoch_{}'.format(epoch))
            self.load_model(model_path=load_from_path)
            batch_statistics = {"train_acc": [], "train_loss": [],"current_epoch": epoch}

            for x_batch, y_batch in valid_set: # (1)
                loss_batch, acc_batch = self.run_evaluation_iter(x_batch,y_batch) # (2)
                batch_statistics["train_loss"].append(loss_batch.item())
                batch_statistics["train_acc"].append(acc_batch)

            statistics_to_save = OrderedDict({"current_epoch":epoch, "eval_acc":0,"eval_loss":0})
            for key,val in batch_statistics.items():
                if key not in ["current_epoch"]:
                    batch_values = np.array(batch_statistics[key])
                    epoch_val = np.mean(batch_values) # mean of all batch values current epooch
                    statistics_to_save[key] = np.around(epoch_val,decimals=4)

            print(statistics_to_save)
            storage_utils.save_statistics(statistics_to_save,eval_results_file_path)

        '''
        Remarks:
        (1) x_batch: array (num_batches,-1), y_batch (num_batches,-1) one-hot.
        (2) loss_batch: tensor(1.9321) type: tensor, acc_batch 0.63 type: numpy.float64.
        '''

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