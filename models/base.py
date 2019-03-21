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
from collections import defaultdict
from attacks.data_augmenter import DataAugmenter
from data_subsetter import DataSubsetter
from data_providers import DataProvider

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

    def get_acc_batch_tens(self,y_batch,y_batch_pred):
        _, y_pred_batch_int = torch.max(y_batch_pred.data, 1)  # argmax of predictions
        acc = np.mean(list(y_pred_batch_int.eq(y_batch.data).cpu()))  # compute accuracy
        return acc

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

    def normal_train(self,train_dataprovider,valid_dataprovider,num_epochs,optimizer,results_dir,scheduler=None):

        if not os.path.exists(results_dir): os.makedirs(results_dir)
        train_results_path = os.path.join(results_dir, 'train_results.txt')
        valid_results_path = os.path.join(results_dir, 'valid_results.txt')
        advers_images_path = os.path.join(results_dir, 'advers_images.pickle')
        model_save_dir = os.path.join(results_dir,'model')

        logger = Logger(stream = sys.stderr,disable= False)

        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        if scheduler is not None:
            self.scheduler = scheduler

        def normal_train_epoch(current_epoch):
            train_statistics_to_save = OrderedDict(
                dict.fromkeys([
                    'current_epoch',
                    'train_acc',
                    'train_loss',
                    'epoch_train_time'
                ])
            )
            epoch_start_time = time.time()
            keys = ['loss_true','acc_true']
            batch_statistics = {key : [] for key in keys}

            for i, (x_batch, y_batch) in tqdm(enumerate(train_dataprovider), file=sys.stderr):  # get data batches
                loss_batch, accuracy_batch = self.train_iter(x_batch, y_batch)
                batch_statistics['loss_true'].append(loss_batch.item())
                batch_statistics['acc_true'].append(accuracy_batch)

            epoch_loss_true = np.mean(np.array(batch_statistics['loss_true']))
            epoch_acc_true = np.mean(np.array(batch_statistics['acc_true']))

            epoch_train_time = time.time() - epoch_start_time
            train_statistics_to_save['current_epoch'] = current_epoch
            train_statistics_to_save['train_acc'] = np.around(epoch_acc_true,decimals=4)
            train_statistics_to_save['train_loss'] = np.around(epoch_loss_true, decimals=4)
            train_statistics_to_save['epoch_train_time'] = epoch_train_time
            return train_statistics_to_save

        def validation_epoch(data,current_epoch): # data is a dataprovider.
            with_replacement = data.with_replacement
            data.with_replacement = False

            batch_statistics = {'loss': [], 'acc': []}
            for i, (x_train_batch, y_train_batch) in tqdm(enumerate(data), file=sys.stderr):  # get data batches
                loss_batch, accuracy_batch = self.run_evaluation_iter(x_train_batch, y_train_batch,integer_encoded=True) # process batch
                batch_statistics['loss'].append(loss_batch.item())
                batch_statistics['acc'].append(accuracy_batch)
            epoch_loss = np.mean(np.array(batch_statistics['loss']))
            epoch_acc = np.mean(np.array(batch_statistics['acc']))

            data.with_replacement = with_replacement # go back to where it was originally - otherwise the advers training get's terminated too early.
            valid_statistics_to_save = OrderedDict({
                'current_epoch': current_epoch,
                'valid_acc': np.around(epoch_acc, decimals=4),
                'valid_loss': np.around(epoch_loss, decimals=4),
            })
            logger.print('finished validating epoch {}'.format(current_epoch))
            return valid_statistics_to_save

        bpm = {'valid_acc': 0}
        torch.cuda.empty_cache()
        for current_epoch in range(self.num_epochs):
            # training.
            train_statistics_to_save = normal_train_epoch(current_epoch) # complete a training epoch.
            storage_utils.save_statistics(train_statistics_to_save, file_path=train_results_path)
            logger.print('finished training epoch {}'.format(current_epoch))
            self.save_model(model_save_dir, model_save_name='model_epoch_{}'.format(str(current_epoch)))

            # validation.
            valid_statistics_to_save = validation_epoch(valid_dataprovider,current_epoch)
            storage_utils.save_statistics(valid_statistics_to_save, file_path=valid_results_path)

            # printing results:
            logger.print(train_statistics_to_save)
            logger.print(valid_statistics_to_save)

            if scheduler is not None: scheduler.step()
            for param_group in self.optimizer.param_groups:
                logger.print('learning rate: {}'.format(param_group['lr']))

    def advers_train_normal(self,
                                  train_dataprovider,
                                  valid_dataprovider,
                                  attack,
                                  num_epochs,
                                  optimizer,
                                  results_dir,
                                  scheduler=None):

        '''
        this function adversarially trains a network the normal way.
        '''

        if not os.path.exists(results_dir): os.makedirs(results_dir)
        train_results_path = os.path.join(results_dir, 'train_results.txt')
        valid_results_path = os.path.join(results_dir, 'valid_results.txt')
        advers_images_path = os.path.join(results_dir, 'advers_images.pickle')
        model_save_dir = os.path.join(results_dir,'model')

        logger = Logger(stream = sys.stderr,disable= False)
        logger.print('starting adversarial training procedure')
        logger.print('attack used: {}'.format(type(attack)))

        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        if scheduler is not None:
            self.scheduler = scheduler

        advs_images_dict = {}

        def advers_train_epoch_normal(current_epoch):
            train_statistics_to_save = OrderedDict(
                dict.fromkeys([
                    'current_epoch',
                    'train_acc',
                    'train_loss',
                    'train_acc_adv',
                    'train_loss_adv',
                    'epoch_train_time'
                ])
            )
            epoch_start_time = time.time()
            keys = ['loss_adv','acc_adv','loss_true','acc_true']
            batch_statistics = {key : [] for key in keys}

            x_batch_adv = None
            for i, (x_batch, y_batch) in tqdm(enumerate(train_dataprovider), file=sys.stderr):  # get data batches
                x_batch_adv = attack(x_batch,y_batch)
                loss_adv, adv_acc_batch, loss_true, true_acc_batch = \
                    self.advers_train_iter_normal(x_batch_adv, x_batch, y_batch)  # process batch

                batch_statistics['loss_adv'].append(loss_adv.item())
                batch_statistics['acc_adv'].append(adv_acc_batch)
                batch_statistics['loss_true'].append(loss_true.item())
                batch_statistics['acc_true'].append(true_acc_batch)

            epoch_loss_adv = np.mean(np.array(batch_statistics['loss_adv']))
            epoch_acc_adv = np.mean(np.array(batch_statistics['acc_adv']))
            epoch_loss_true = np.mean(np.array(batch_statistics['loss_true']))
            epoch_acc_true = np.mean(np.array(batch_statistics['acc_true']))
            attack.model = self # updating model of attack!
            epoch_train_time = time.time() - epoch_start_time

            train_statistics_to_save['current_epoch'] = current_epoch
            train_statistics_to_save['train_acc'] = np.around(epoch_acc_true,decimals=4)
            train_statistics_to_save['train_loss'] = np.around(epoch_loss_true, decimals=4)
            train_statistics_to_save['train_acc_adv'] = np.around(epoch_acc_adv, decimals=4)
            train_statistics_to_save['train_loss_adv'] = np.around(epoch_loss_adv, decimals=4)
            train_statistics_to_save['epoch_train_time'] = epoch_train_time
            advs_images_dict[current_epoch] = x_batch_adv
            return train_statistics_to_save

        def validation_epoch(data,current_epoch): # data is a dataprovider.
            with_replacement = data.with_replacement
            data.with_replacement = False

            batch_statistics = {'loss': [], 'acc': []}
            for i, (x_train_batch, y_train_batch) in tqdm(enumerate(data), file=sys.stderr):  # get data batches
                loss_batch, accuracy_batch = self.run_evaluation_iter(x_train_batch, y_train_batch,integer_encoded=True) # process batch
                batch_statistics['loss'].append(loss_batch.item())
                batch_statistics['acc'].append(accuracy_batch)
            epoch_loss = np.mean(np.array(batch_statistics['loss']))
            epoch_acc = np.mean(np.array(batch_statistics['acc']))

            data.with_replacement = with_replacement # go back to where it was originally - otherwise the advers training get's terminated too early.
            valid_statistics_to_save = OrderedDict({
                'current_epoch': current_epoch,
                'valid_acc': np.around(epoch_acc, decimals=4),
                'valid_loss': np.around(epoch_loss, decimals=4),
            })
            logger.print('finished validating epoch {}'.format(current_epoch))
            return valid_statistics_to_save

        import pickle
        bpm = {'valid_acc': 0}
        torch.cuda.empty_cache()
        for current_epoch in range(self.num_epochs):
            # training.
            train_statistics_to_save = advers_train_epoch_normal(current_epoch) # complete a training epoch.
            storage_utils.save_statistics(train_statistics_to_save, file_path=train_results_path)
            with open(advers_images_path, 'wb') as f: # note you overwrite the file each time but that okay since advs_images_dict grows each epoch.
                pickle.dump(advs_images_dict, f)
            logger.print('finished training epoch {}'.format(current_epoch))
            logger.print('finished saving adversarial images epoch {}'.format(current_epoch))
            self.save_model(model_save_dir, model_save_name='model_epoch_{}'.format(str(current_epoch)))

            # validation.
            valid_statistics_to_save = validation_epoch(valid_dataprovider,current_epoch)
            storage_utils.save_statistics(valid_statistics_to_save, file_path=valid_results_path)

            # printing results:
            logger.print(train_statistics_to_save)
            logger.print(valid_statistics_to_save)

            if scheduler is not None: scheduler.step()
            for param_group in self.optimizer.param_groups:
                logger.print('learning rate: {}'.format(param_group['lr']))

    def advers_train_and_evaluate_uniform_tens(self, train_sampler, valid_full, test_full, attack, num_epochs,
                                          optimizer, results_dir,
                                          scheduler=None, minority_class=3):

        if not os.path.exists(results_dir): os.makedirs(results_dir)
        train_results_path = os.path.join(results_dir, 'train_results.txt')
        valid_and_test_results_path = os.path.join(results_dir, 'valid_and_test_results.txt')
        advers_images_path = os.path.join(results_dir, 'advers_images.pickle')
        model_save_dir = os.path.join(results_dir, 'model')

        logger = Logger(stream=sys.stderr, disable=False)
        logger.print('starting adversarial training procedure')
        logger.print('attack used: {}'.format(type(attack)))

        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        if scheduler is not None:
            self.scheduler = scheduler

        import pickle
        advs_images_dict = {}

        def training_epoch(current_epoch):  # done i think!
            batch_statistics = defaultdict(lambda: [])
            epoch_start_time = time.time()
            with tqdm(total=len(train_sampler)) as pbar_train:
                for i, batch in enumerate(train_sampler):

                    (x_maj_batch, y_maj_batch, x_min_batch, y_min_batch) = batch

                    x_maj_batch = x_maj_batch.float().to(device=self.device)
                    y_maj_batch = y_maj_batch.long().to(device=self.device)


                    # print(x_maj_batch.shape, y_maj_batch.shape, x_min_batch.shape, y_min_batch.shape)

                    if x_min_batch is not None:
                        # logger.print("START ATTACK.")
                        x_min_batch = x_min_batch.float().to(device=self.device)
                        y_min_batch = y_min_batch.long().to(device=self.device)
                        start_attack = time.time()
                        #x_min_batch_adv = attack(x_min_batch, y_min_batch)
                        # logger.print("END ATTACK. TOOK: {}".format(time.time() - start_attack))

                        x_min_batch_adv = x_min_batch

                        # from data_viewer import ImageDataViewer
                        # print(x_mino_batch_adv.shape)
                        # print(x_mino_batch_adv)
                        # ImageDataViewer.batch_view(np.array(x_mino_batch_adv), cmap=None,nrows=3,ncols=2,labels=[i for i in range(6)],hspace=0,wspace=0)
                        # exit()
                        #
                        # x_comb_batch = torch.cat([x_maj_batch,x_mino_batch,x_mino_batch_adv],dim=0)
                        # y_comb_batch = torch.cat([y_maj_batch, y_mino_batch, y_mino_batch], dim=0)
                        x_comb_batch = torch.cat([x_maj_batch, x_min_batch, x_min_batch_adv], dim=0)
                        y_comb_batch = torch.cat([y_maj_batch, y_min_batch, y_min_batch], dim=0)

                    else:
                        x_comb_batch = x_maj_batch
                        y_comb_batch = y_maj_batch

                    start_train = time.time()
                    # logger.print("START TTRAINING ITER.")
                    print(x_comb_batch.shape, y_comb_batch.shape, x_min_batch_adv.shape, y_min_batch.shape)

                    loss_comb, accuracy_comb, loss_mino_adv, acc_mino_adv = \
                        self.train_iter_advers_tens(x_comb_batch, y_comb_batch, x_adv=x_min_batch_adv, y_adv=y_min_batch)  # process batch

                    # logger.print("END TRAINING ITER. TOOK: {}".format(time.time() - start_train))

                    if loss_mino_adv is not None:
                        batch_statistics['train_loss_mino_adv'].append(loss_mino_adv.item())
                        batch_statistics['train_acc_mino_adv'].append(acc_mino_adv)
                    batch_statistics['train_loss_comb'].append(loss_comb.item())
                    batch_statistics['train_acc_comb'].append(accuracy_comb)
                    string_description = " ".join(["{}:{:.4f}".format(key, np.mean(value)) for key, value in batch_statistics.items()])
                    pbar_train.update(1)
                    pbar_train.set_description(string_description)

            epoch_stats = OrderedDict({})
            epoch_train_time = time.time() - epoch_start_time
            epoch_stats['current_epoch'] = current_epoch
            for k, v in batch_statistics.items():
                epoch_stats[k] = np.around(np.mean(v), decimals=4)
            epoch_stats['epoch_train_time'] = epoch_train_time

            attack.model = self  # updating model of attack!
            if x_min_batch_adv is not None:
                advs_images_dict[current_epoch] = x_min_batch_adv.detach().clone().cpu().numpy()
            else:
                advs_images_dict[current_epoch] = None

            train_statistics_to_save = epoch_stats
            return train_statistics_to_save

        def test_epoch(current_epoch):  # done i think.
            batch_statistics = defaultdict(lambda: [])

            with tqdm(total=len(valid_full)) as pbar_val:
                for i, batch in enumerate(valid_full):
                    x_all, y_all = batch
                    x_all = x_all.to(device=self.device)
                    y_all = y_all.to(device=self.device)

                    output = self.run_evaluation_iter(x_all, y_all, integer_encoded=True, minority_class=minority_class)
                    batch_statistics['valid_loss'].append(output['loss'].item())
                    batch_statistics['valid_acc'].append(output['acc'])
                    if output['loss_min'] is not None:
                        batch_statistics['valid_loss_minority'].append(output['loss_min'].item())
                        batch_statistics['valid_acc_minority'].append(output['acc_min'])
                    string_description = " ".join(["{}:{:.4f}".format(key, np.mean(value)) for key, value in batch_statistics.items()])
                    pbar_val.update(1)
                    pbar_val.set_description(string_description)

            with tqdm(total=len(test_full)) as pbar_test:
                for i, batch in enumerate(test_full):
                    x_all, y_all = batch
                    x_all = x_all.to(device=self.device)
                    y_all = y_all.to(device=self.device)

                    output = self.run_evaluation_iter(x_all, y_all, integer_encoded=True, minority_class=minority_class)
                    batch_statistics['test_loss'].append(output['loss'].item())
                    batch_statistics['test_acc'].append(output['acc'])
                    if output['loss_min'] is not None:
                        batch_statistics['test_loss_minority'].append(output['loss_min'].item())
                        batch_statistics['test_acc_minority'].append(output['acc_min'])
                    string_description = " ".join(
                        ["{}: {:.4f}".format(key, np.mean(value)) for key, value in batch_statistics.items()])
                    pbar_test.update(1)
                    pbar_test.set_description(string_description)


            epoch_stats = OrderedDict({})
            epoch_stats['current_epoch'] = current_epoch
            for k, v in batch_statistics.items():
                epoch_stats[k] = np.around(np.mean(v), decimals=4)

            test_statistics_to_save = epoch_stats
            return test_statistics_to_save

        bpm = defaultdict(lambda: 0)
        torch.cuda.empty_cache()
        for current_epoch in range(self.num_epochs):
            train_statistics_to_save = training_epoch(current_epoch)

            # save train statistics.
            storage_utils.save_statistics(train_statistics_to_save, file_path=train_results_path)

            # save adversarial images.
            with open(advers_images_path,
                      'wb') as f:  # note you overwrite the file each time but that okay since advs_images_dict grows each epoch.
                pickle.dump(advs_images_dict, f)

            # save model.
            self.save_model(model_save_dir, model_save_name='model_epoch_{}'.format(str(current_epoch)))
            logger.print(train_statistics_to_save)

            # test performance.
            test_statistics_to_save = test_epoch(current_epoch)
            valid_acc_all = test_statistics_to_save['valid_acc']
            valid_acc_mino = test_statistics_to_save['valid_acc_minority']

            if valid_acc_all > bpm['valid_acc_all']:
                bpm['valid_acc_all'] = valid_acc_all
                bpm['test_acc_all'] = test_statistics_to_save['test_acc']
                bpm['best_epoch_all'] = current_epoch

            if valid_acc_mino > bpm['valid_acc_mino']:
                bpm['valid_acc_mino'] = valid_acc_mino
                bpm['test_acc_mino'] = test_statistics_to_save['test_acc_minority']
                bpm['best_epoch_mino'] = current_epoch

            test_statistics_to_save['bpm_epoch_all'] = bpm['best_epoch_all']
            test_statistics_to_save['bpm_valid_acc_all'] = bpm['valid_acc_all']
            test_statistics_to_save['bpm_test_acc_all'] = bpm['test_acc_all']
            test_statistics_to_save['bpm_epoch_mino'] = bpm['best_epoch_mino']
            test_statistics_to_save['bpm_valid_acc_mino'] = bpm['valid_acc_mino']
            test_statistics_to_save['bpm_test_acc_mino'] = bpm['test_acc_mino']

            storage_utils.save_statistics(test_statistics_to_save, file_path=valid_and_test_results_path)
            logger.print(test_statistics_to_save)

            if scheduler is not None: scheduler.step()
            for param_group in self.optimizer.param_groups:
                logger.print('learning rate: {}'.format(param_group['lr']))



    def advers_train_and_evaluate_uniform(self,train_sampler,valid_full,test_full,attack,num_epochs,optimizer,results_dir,
                                          scheduler=None):

        if not os.path.exists(results_dir): os.makedirs(results_dir)
        train_results_path = os.path.join(results_dir, 'train_results.txt')
        valid_and_test_results_path = os.path.join(results_dir, 'valid_and_test_results.txt')
        advers_images_path = os.path.join(results_dir, 'advers_images.pickle')
        model_save_dir = os.path.join(results_dir,'model')

        logger = Logger(stream = sys.stderr,disable= False)
        logger.print('starting adversarial training procedure')
        logger.print('attack used: {}'.format(type(attack)))

        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        if scheduler is not None:
            self.scheduler = scheduler

        import pickle
        advs_images_dict = {}

        def training_epoch(current_epoch): # done i think!
            batch_statistics = defaultdict(lambda : [])

            epoch_start_time = time.time()
            x_mino_batch_adv = None

            for i, batch in tqdm(enumerate(train_sampler),file=sys.stderr):
                (x_maj_batch, y_maj_batch, x_mino_batch, y_mino_batch) = batch


                if len(x_mino_batch) > 0: # it's possible to not sample any from the minority.
                    start_attack = time.time()
                    logger.print("START ATTACK.")
                    x_mino_batch_adv = attack(x_mino_batch,y_mino_batch)
                    logger.print("END ATTACK. TOOK: {}".format(time.time()-start_attack))
                    #x_mino_batch_adv = x_mino_batch_adv.detach().numpy()

                    if x_maj_batch is not None: # why is this necessary?
                        # x_comb_batch = torch.cat([x_maj_batch,x_mino_batch,x_mino_batch_adv],dim=0)
                        x_comb_batch = np.vstack((x_maj_batch,x_mino_batch,x_mino_batch_adv))
                        y_comb_batch = np.hstack((y_maj_batch,y_mino_batch,y_mino_batch))
                    else:
                        x_comb_batch = np.vstack((x_mino_batch, x_mino_batch_adv))
                        y_comb_batch = np.hstack((y_mino_batch, y_mino_batch))
                else:
                    x_comb_batch = x_maj_batch
                    y_comb_batch = y_maj_batch
                    x_mino_batch_adv = None
                    y_mino_batch = None

                start_train = time.time()
                logger.print("START TTRAINING ITER.")
                loss_comb, accuracy_comb, loss_mino_adv, acc_mino_adv = \
                    self.train_iter_advers(x_comb_batch,y_comb_batch,x_mino_batch_adv,y_mino_batch)  # process batch
                logger.print("END TRAINING ITER. TOOK: {}".format(time.time()-start_train))

                if loss_mino_adv is not None:
                    batch_statistics['train_loss_mino_adv'].append(loss_mino_adv.item())
                    batch_statistics['train_acc_mino_adv'].append(acc_mino_adv)
                batch_statistics['train_loss_comb'].append(loss_comb.item())
                batch_statistics['train_acc_comb'].append(accuracy_comb)

            # epoch_stats = OrderedDict.fromkeys([
            #     'current_epoch','train_loss_mino_adv','train_acc_mino_adv','train_loss_comb','train_acc_comb',
            #     'epoch_train_time'
            # ])
            epoch_stats = OrderedDict({})
            epoch_train_time = time.time() - epoch_start_time
            epoch_stats['current_epoch'] = current_epoch
            for k, v in batch_statistics.items():
                epoch_stats[k] = np.around(np.mean(v),decimals=4)
            epoch_stats['epoch_train_time'] = epoch_train_time

            attack.model = self  # updating model of attack!
            advs_images_dict[current_epoch] = x_mino_batch_adv.detach().clone().cpu().numpy()
            train_statistics_to_save = epoch_stats
            return train_statistics_to_save

        def test_epoch(current_epoch): # done i think.
            batch_statistics = defaultdict(lambda: [])

            for i, batch in tqdm(enumerate(valid_sampler.full_sampler),file=sys.stderr):
                x_all, y_all = batch
                loss_all, acc_all = self.run_evaluation_iter(x_all, y_all, integer_encoded=True)
                batch_statistics['valid_loss'].append(loss_all.item())
                batch_statistics['valid_acc'].append(acc_all)

            for i, batch in tqdm(enumerate(valid_sampler.mino_sampler),file=sys.stderr):
                x_mino, y_mino = batch
                loss_mino, acc_mino = self.run_evaluation_iter(x_mino, y_mino, integer_encoded=True)
                batch_statistics['valid_loss_mino'].append(loss_mino.item())
                batch_statistics['valid_acc_mino'].append(acc_mino)

            for i, batch in tqdm(enumerate(test_sampler.full_sampler),file=sys.stderr):
                x_all, y_all = batch
                loss_all, acc_all = self.run_evaluation_iter(x_all, y_all, integer_encoded=True)
                batch_statistics['test_loss'].append(loss_all.item())
                batch_statistics['test_acc'].append(acc_all)

            for i, batch in tqdm(enumerate(test_sampler.mino_sampler),file=sys.stderr):
                x_mino, y_mino = batch
                loss_mino, acc_mino = self.run_evaluation_iter(x_mino, y_mino, integer_encoded=True)
                batch_statistics['test_loss_mino'].append(loss_mino.item())
                batch_statistics['test_acc_mino'].append(acc_mino)
                exit()

            epoch_stats = OrderedDict({})
            epoch_stats['current_epoch'] = current_epoch
            for k, v in batch_statistics.items():
                epoch_stats[k] = np.around(np.mean(v),decimals=4)

            test_statistics_to_save = epoch_stats
            return test_statistics_to_save

        bpm = defaultdict(lambda : 0)
        torch.cuda.empty_cache()
        for current_epoch in range(self.num_epochs):
            train_statistics_to_save = training_epoch(current_epoch)

            # save train statistics.
            storage_utils.save_statistics(train_statistics_to_save, file_path=train_results_path)

            # save adversarial images.
            with open(advers_images_path,'wb') as f:  # note you overwrite the file each time but that okay since advs_images_dict grows each epoch.
                pickle.dump(advs_images_dict, f)

            # save model.
            self.save_model(model_save_dir, model_save_name='model_epoch_{}'.format(str(current_epoch)))
            logger.print(train_statistics_to_save)

            # test performance.
            test_statistics_to_save = test_epoch(current_epoch)
            valid_acc_all = test_statistics_to_save['valid_acc']
            valid_acc_mino = test_statistics_to_save['valid_acc_mino']

            if valid_acc_all > bpm['valid_acc_all']:
                bpm['valid_acc_all'] = valid_acc_all
                bpm['test_acc_all'] = test_statistics_to_save['test_acc']
                bpm['best_epoch_all'] = current_epoch

            if valid_acc_mino > bpm['valid_acc_mino']:
                bpm['valid_acc_mino'] = valid_acc_mino
                bpm['test_acc_mino'] = test_statistics_to_save['test_acc_mino']
                bpm['best_epoch_mino'] = current_epoch

            test_statistics_to_save['bpm_epoch_all'] = bpm['best_epoch_all']
            test_statistics_to_save['bpm_valid_acc_all'] = bpm['valid_acc_all']
            test_statistics_to_save['bpm_test_acc_all'] = bpm['test_acc_all']
            test_statistics_to_save['bpm_epoch_mino'] = bpm['best_epoch_mino']
            test_statistics_to_save['bpm_valid_acc_mino'] = bpm['valid_acc_mino']
            test_statistics_to_save['bpm_test_acc_mino'] = bpm['test_acc_mino']

            storage_utils.save_statistics(test_statistics_to_save,file_path=valid_and_test_results_path)
            logger.print(test_statistics_to_save)

            if scheduler is not None: scheduler.step()
            for param_group in self.optimizer.param_groups:
                logger.print('learning rate: {}'.format(param_group['lr']))

    def advers_train_and_evaluate(self,
                                  train_majority_dataprovider,
                                  train_minority_dataprovider,
                                  valid_dataprovider,
                                  valid_minority_dataprovider,
                                  attack,
                                  num_epochs,
                                  optimizer,
                                  results_dir,
                                  scheduler=None):

        if not os.path.exists(results_dir): os.makedirs(results_dir)
        train_results_path = os.path.join(results_dir, 'train_results.txt')
        valid_results_path = os.path.join(results_dir, 'valid_results.txt')
        advers_images_path = os.path.join(results_dir, 'advers_images.pickle')
        model_save_dir = os.path.join(results_dir,'model')

        logger = Logger(stream = sys.stderr,disable= False)
        logger.print('starting adversarial training procedure')
        logger.print('attack used: {}'.format(type(attack)))

        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        if scheduler is not None:
            self.scheduler = scheduler


        def advers_train_epoch():
            batch_statistics = {'loss': [], 'acc': []}
            xm_batch_adv = None

            for i, (xo_batch, yo_batch) in tqdm(enumerate(train_majority_dataprovider), file=sys.stderr):  # get data batches
                xm_batch, ym_batch = train_minority_dataprovider.__next__()
                xm_batch_adv = attack(xm_batch,ym_batch)  # DataAugmenter.advers_attack(xm_batch, ym_batch, attack=attack, disable_progress=disable_progress)
                xm_batch_comb = np.vstack((xo_batch,xm_batch,xm_batch_adv))
                ym_batch_comb = np.hstack((yo_batch,ym_batch,ym_batch))
                loss_batch, accuracy_batch = self.train_iter(xm_batch_comb, ym_batch_comb)  # process batch
                batch_statistics['loss'].append(loss_batch.item())
                batch_statistics['acc'].append(accuracy_batch)

            epoch_loss = np.mean(np.array(batch_statistics['loss']))
            epoch_acc = np.mean(np.array(batch_statistics['acc']))

            attack.model = self
            # print("epoch ended. updated model of attack. ")

            return epoch_loss, epoch_acc, xm_batch_adv

        def validation_epoch(data):
            '''
            :param func: the function that determines how batches are processed. either self.train_iter, or
            self.evaluation_iter. if func==self.train_iter data is being processed for the purpose of updating the
            weights of the network.
            :param data: DataProvider object.
            :return epoch accuracy and loss.
            '''

            with_replacement = data.with_replacement
            data.with_replacement = False

            batch_statistics = {'loss': [], 'acc': []}
            for i, (x_train_batch, y_train_batch) in tqdm(enumerate(data), file=sys.stderr):  # get data batches
                loss_batch, accuracy_batch = self.run_evaluation_iter(x_train_batch, y_train_batch,integer_encoded=True) # process batch
                batch_statistics['loss'].append(loss_batch.item())
                batch_statistics['acc'].append(accuracy_batch)

            epoch_loss = np.mean(np.array(batch_statistics['loss']))
            epoch_acc = np.mean(np.array(batch_statistics['acc']))

            data.with_replacement = with_replacement # go back to where it was originally - otherwise the advers training get's terminated too early.
            return epoch_loss, epoch_acc

        advs_images_dict = {}

        bpm = {'valid_acc': 0}
        torch.cuda.empty_cache()
        for current_epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            train_epoch_loss, train_epoch_acc, xm_batch_adv = advers_train_epoch()

            advs_images_dict[current_epoch] = xm_batch_adv # save batch of images as results.
            epoch_train_time = time.time() - epoch_start_time

            import pickle
            with open(advers_images_path, 'wb') as f: # note you overwrite the file each time but that okay since advs_images_dict grows each epoch.
                pickle.dump(advs_images_dict, f)

            logger.print('finished training epoch {}'.format(current_epoch))
            logger.print('finished saving adversarial images epoch {}'.format(current_epoch))

            self.save_model(model_save_dir, model_save_name='model_epoch_{}'.format(str(current_epoch)))

            valid_epoch_loss, valid_epoch_acc = validation_epoch(data=valid_dataprovider)
            target_valid_epoch_loss, target_valid_epoch_acc = validation_epoch(data=valid_minority_dataprovider)
            target_train_epoch_loss, target_train_epoch_acc = validation_epoch(data=train_minority_dataprovider)

            train_statistics_to_save = OrderedDict({
                'current_epoch': current_epoch,
                'train_acc': np.around(train_epoch_acc, decimals=4),  # round results to 4 decimals.
                'train_loss': np.around(train_epoch_loss, decimals=4),
                'target_train_acc': np.around(target_train_epoch_acc,decimals=4),
                'target_train_loss': np.around(target_train_epoch_loss,decimals=4),
                'epoch_train_time': epoch_train_time,
            })
            storage_utils.save_statistics(train_statistics_to_save, file_path=train_results_path)

            valid_statistics_to_save = OrderedDict({
                'current_epoch': current_epoch,
                'valid_acc': np.around(valid_epoch_acc, decimals=4),
                'valid_loss': np.around(valid_epoch_loss, decimals=4),
                'target_valid_acc': np.around(target_valid_epoch_acc, decimals=4),
                'target_valid_loss': np.around(target_valid_epoch_loss, decimals=4),
            })

            if valid_epoch_acc > bpm['valid_acc']:
                bpm['valid_acc'] = valid_epoch_acc
                bpm['train_acc'] = train_epoch_acc
                bpm['epoch'] = current_epoch
                bpm['train_loss'] = train_epoch_loss
                bpm['valid_loss'] = valid_epoch_loss

            storage_utils.save_statistics(valid_statistics_to_save, file_path=valid_results_path)

            results_to_print = {
                'epoch': current_epoch,
                'best_valid_acc': bpm['valid_acc'],
                'valid_acc': valid_epoch_acc,
                'target_valid_acc': target_valid_epoch_acc,
                'train_acc': train_epoch_acc,
                'valid_loss': valid_epoch_loss,
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

            for i, (x_train_batch, y_train_batch) in tqdm(enumerate(data), file=sys.stderr):  # get data batches
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

    def advers_train_iter_normal(self, x_adv_train_batch, x_true_train_batch, y_train_batch):
        self.train()
        criterion = nn.CrossEntropyLoss().cuda()
        y_train_batch_int = np.int64(y_train_batch.reshape(-1, )) # integer encoded.

        y_train_batch_int = torch.Tensor(y_train_batch_int).long().to(device=self.device)
        x_adv_train_batch = torch.Tensor(x_adv_train_batch).float().to(device=self.device)
        y_adv_pred_batch = self.forward(x_adv_train_batch)  # model forward pass
        loss_adv = criterion(input=y_adv_pred_batch,
                         target=y_train_batch_int)  # self.cross_entropy(input=y_pred_batch,target=y_train_batch_int)

        self.optimizer.zero_grad()
        loss_adv.backward()
        self.optimizer.step()

        adv_acc_batch = self.get_acc_batch(x_adv_train_batch, y_train_batch, y_adv_pred_batch, integer_encoded=True)

        x_true_train_batch = torch.Tensor(x_true_train_batch).float().to(device=self.device)
        y_true_pred_batch = self.forward(x_true_train_batch)
        loss_true = criterion(input=y_true_pred_batch,target=y_train_batch_int)
        true_acc_batch = self.get_acc_batch(x_true_train_batch, y_train_batch, y_true_pred_batch, integer_encoded=True)

        return loss_adv.data, adv_acc_batch, loss_true.data, true_acc_batch


    # def train_iter_advers_tens(self,x_comb,y_comb,x_adv,y_adv):
    #     self.train()
    #     criterion = nn.CrossEntropyLoss().cuda()
    #
    #     y_pred_comb = self.forward(x_comb)
    #
    #     loss_comb = criterion(input=y_pred_comb,target=y_comb.view(-1))
    #     self.optimizer.zero_grad()
    #     loss_comb.backward()
    #     self.optimizer.step()
    #     acc_comb_batch = self.get_acc_batch_tens(y_comb, y_pred_comb)
    #
    #     if x_adv is not None: # happens when batch doesn't have minority data in it.
    #         y_pred_adv = self.forward(x_adv)
    #         loss_adv = criterion(input=y_pred_adv, target=y_adv.view(-1))
    #         acc_adv = self.get_acc_batch_tens(y_adv,y_pred_adv)
    #         output = (loss_comb.data, acc_comb_batch, loss_adv.data, acc_adv)
    #     else:
    #         output = (loss_comb.data, acc_comb_batch, None, None)
    #
    #     return output

    def train_iter_advers_tens(self,x_comb, y_comb, x_adv=None, y_adv=None):
        self.train()
        criterion = nn.CrossEntropyLoss().cuda()

        y_pred_comb = self.forward(x_comb)

        loss_comb = criterion(input=y_pred_comb,target=y_comb.view(-1))
        self.optimizer.zero_grad()
        loss_comb.backward()
        self.optimizer.step()
        acc_comb_batch = self.get_acc_batch_tens(y_comb, y_pred_comb)

        if x_adv is not None: # happens when batch doesn't have minority data in it.
            y_pred_adv = self.forward(x_adv)
            loss_adv = criterion(input=y_pred_adv, target=y_adv.view(-1))
            acc_adv = self.get_acc_batch_tens(y_adv,y_pred_adv)
            output = (loss_comb.data, acc_comb_batch, loss_adv.data, acc_adv)
        else:
            output = (loss_comb.data, acc_comb_batch, None, None)

        return output


    def train_iter_advers(self,x_train_comb_batch,y_train_batch,x_train_adv_batch,y_train_adv_batch):
        self.train()
        criterion = nn.CrossEntropyLoss().cuda()
        y_train_comb_batch_int = np.int64(y_train_batch.reshape(-1, ))
        if y_train_adv_batch is not None:
            y_train_adv_batch_int = np.int64(y_train_adv_batch.reshape(-1,))
            y_train_adv_batch_int = torch.Tensor(y_train_adv_batch_int).long().to(device=self.device)

        y_train_comb_batch_int = torch.Tensor(y_train_comb_batch_int).long().to(device=self.device)
        x_train_comb_batch = torch.Tensor(x_train_comb_batch).float().to(device=self.device)
        y_pred_comb_batch = self.forward(x_train_comb_batch)  # model forward pass
        loss_comb = criterion(input=y_pred_comb_batch,
                         target=y_train_comb_batch_int)  # self.cross_entropy(input=y_pred_batch,target=y_train_batch_int)
        self.optimizer.zero_grad()
        loss_comb.backward()
        self.optimizer.step()
        acc_comb_batch = self.get_acc_batch(x_train_comb_batch, y_train_batch, y_pred_comb_batch, integer_encoded=True)

        x_train_adv_batch = None

        if x_train_adv_batch is not None:
            x_train_adv_batch = torch.Tensor(x_train_adv_batch).float().to(device=self.device)
            y_pred_adv_batch = self.forward(x_train_adv_batch)
            loss_adv = criterion(input=y_pred_adv_batch,target=y_train_adv_batch_int)
            acc_adv_batch = self.get_acc_batch(x_train_adv_batch,y_train_adv_batch,y_pred_adv_batch,integer_encoded=True)

            output = (loss_comb.data, acc_comb_batch, loss_adv.data, acc_adv_batch)
        else:
            output = (loss_comb.data, acc_comb_batch, None, None)

        return output



    def train_iter(self, x_train_batch, y_train_batch, integer_encoded=True):
        """
        :param x_train_batch: array
        :param y_train_batch: array, one-hot-encoded or integer encoded.
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
        loss = criterion(input=y_pred_batch,target=y_train_batch_int) # self.cross_entropy(input=y_pred_batch,target=y_train_batch_int)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        acc_batch = self.get_acc_batch(x_train_batch,y_train_batch,y_pred_batch,integer_encoded=integer_encoded)

        return loss.data, acc_batch

    def run_evaluation_iter(self, x_batch, y_batch, PRINTFLAG = False, integer_encoded=False, minority_class=3):
        '''
        :param x_batch:
        :param y_batch:
        :return:
        '''
        with torch.no_grad():
            self.eval()
            preds = self.forward(x_batch)  # model forward pass

            x_min = []
            y_min = []
            y_min_preds = []

            for i in range(y_batch.shape[0]):
                if int(y_batch[i].data) == minority_class:
                    x_min.append(x_batch[i])
                    y_min.append(y_batch[i])
                    y_min_preds.append(preds[i])

            loss_batch = F.cross_entropy(input=preds, target=y_batch)
            acc_batch = self.get_acc_batch(x_batch.data.cpu().numpy(), y_batch.data.cpu().numpy(), preds,
                                           integer_encoded=integer_encoded)

            if len(x_min) > 0:
                x_min = torch.stack(x_min, dim=0).to(device=self.device)
                y_min = torch.stack(y_min, dim=0).to(device=self.device)
                y_min_preds = torch.stack(y_min_preds, dim=0).to(device=self.device)

                loss_min = F.cross_entropy(input=y_min_preds, target=y_min)
                acc_min = self.get_acc_batch(x_min.data.cpu().numpy(), y_min.data.cpu().numpy(),
                                             integer_encoded=integer_encoded)

                output = {'loss': loss_batch.data, 'acc': acc_batch, 'loss_min': loss_min.data, 'acc_min': acc_min}

            else:
                output = {'loss': loss_batch.data, 'acc': acc_batch, 'loss_min': None, 'acc_min': None}




        return output

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