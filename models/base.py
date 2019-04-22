import numpy as np
import torch
import os
from models import storage_utils
from tqdm import tqdm
import sys
from collections import OrderedDict
import torch.nn as nn
from collections import defaultdict
import pickle


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

        logger = Logger(stream=sys.stderr)
        logger.disable = False # if disabled does not print info messages.
        logger.module_name = __file__
        gpu_ids = '0'
        if not torch.cuda.is_available():
            print("GPU IS NOT AVAILABLE")
            self.device = torch.device('cpu')  # sets the device to be CPU

        if ',' in gpu_ids:
            self.device = [torch.device('cuda:{}'.format(idx)) for idx in gpu_ids.split(",")]
        else:
            self.device = torch.device('cuda:{}'.format(int(gpu_ids)))

        if type(self.device) is list:
            self.device = self.device[0]

        logger.print("gpu ids being used: {}".format(gpu_ids))

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids  # (1)
        self.cuda() # (2)

        '''
        remarks:
        (1) sets the main GPU to be the one at index 0 (on multi gpu machines you can choose which one you want to use 
        by using the relevant GPU ID)
        (2) this makes model a cuda model. makes it so that gpu is used with it.
        '''

    def get_acc_batch(self, y_batch, y_batch_pred):
        _, y_pred_batch_int = torch.max(y_batch_pred.data, 1)  # argmax of predictions
        acc = np.mean(list(y_pred_batch_int.eq(y_batch.data).cpu()))  # compute accuracy
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

    def train_evaluate(self, train_set, valid_full, test_full, num_epochs,
                       optimizer, results_dir,
                       minority_class,
                       attack=None,
                       scheduler=None):

        # SET OUTPUT PATH
        if not os.path.exists(results_dir): os.makedirs(results_dir)
        train_results_path = os.path.join(results_dir, 'train_results.txt')
        valid_and_test_results_path = os.path.join(results_dir, 'valid_and_test_results.txt')
        model_save_dir = os.path.join(results_dir, 'model')

        if attack is not None:
            advers_images_path = os.path.join(results_dir, 'advers_images.pickle')
            advs_images_dict = {}

        # SET LOGS
        logger = Logger(stream=sys.stderr, disable=False)
        logger.print('starting adversarial training procedure')
        logger.print('attack used: {}'.format(type(attack) if attack is not None else 'None'))

        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        if scheduler is not None:
            self.scheduler = scheduler

        def train_epoch(current_epoch):
            batch_statistics = defaultdict(lambda: [])
            with tqdm(total=len(train_set)) as pbar_val:
                for i, batch in enumerate(train_set):
                    x, y = batch
                    x = x.to(device=self.device)
                    y = y.to(device=self.device)

                    output = self.train_iteration(x, y, minority_class)

                    # SAVE BATCH STATS
                    for key, value, in output.items():
                        batch_statistics[key].append(output[key].item())

                    # SET PBAR
                    string_description = " ".join(
                        ["{}:{:.4f}".format(key, np.mean(value)) for key, value in batch_statistics.items()])
                    pbar_val.update(1)
                    # pbar_val.set_description(string_description)

            epoch_stats = OrderedDict({})
            epoch_stats['current_epoch'] = current_epoch
            for k, v in batch_statistics.items():
                epoch_stats[k] = np.around(np.mean(v), decimals=4)
            return epoch_stats
        #
        #
        # def training_epoch_adversarial(current_epoch):
        #     batch_statistics = defaultdict(lambda: [])
        #     epoch_start_time = time.time()
        #     with tqdm(total=len(train_sampler)) as pbar_train:
        #         for i, batch in enumerate(train_sampler):
        #
        #             (x_maj_batch, y_maj_batch, x_min_batch, y_min_batch) = batch
        #
        #             x_maj_batch = x_maj_batch.float().to(device=self.device)
        #             y_maj_batch = y_maj_batch.long().to(device=self.device)
        #             x_min_batch_adv = x_min_batch
        #             if x_min_batch is not None:
        #                 x_min_batch = x_min_batch.float().to(device=self.device)
        #                 y_min_batch = y_min_batch.long().to(device=self.device)
        #                 if attack is not None:
        #                     x_min_batch_adv = attack(x_min_batch, y_min_batch)
        #
        #                 x_comb_batch = torch.cat([x_maj_batch, x_min_batch, x_min_batch_adv], dim=0)
        #                 y_comb_batch = torch.cat([y_maj_batch, y_min_batch, y_min_batch], dim=0)
        #                 y_min_map = (y_maj_batch.shape[0], y_maj_batch.shape[0] + y_min_batch.shape[0])
        #                 y_min_adv_map = (y_maj_batch.shape[0] + y_min_batch.shape[0], y_comb_batch.shape[0])
        #
        #             else:
        #                 x_comb_batch = x_maj_batch
        #                 y_comb_batch = y_maj_batch
        #                 y_min_batch = None
        #                 y_min_map = (None)
        #                 y_min_adv_map = (None)
        #
        #             loss_comb, accuracy_comb, loss_min, acc_min, loss_mino_adv, acc_mino_adv = \
        #                 self.train_iteration(x_comb_batch, y_comb_batch, y_min_map=y_min_map, y_min_adv_map=y_min_adv_map, x_adv=x_min_batch_adv, y_adv=y_min_batch)  # process batch
        #
        #             if loss_mino_adv is not None:
        #                 batch_statistics['train_loss_min_adv'].append(loss_mino_adv.item())
        #                 batch_statistics['train_acc_min_adv'].append(acc_mino_adv)
        #             if loss_min is not None:
        #                 batch_statistics['train_loss_min'].append(loss_min.item())
        #                 batch_statistics['train_acc_min'].append(acc_min)
        #
        #             batch_statistics['train_loss_comb'].append(loss_comb.item())
        #             batch_statistics['train_acc_comb'].append(accuracy_comb)
        #             string_description = " ".join(["{}:{:.4f}".format(key, np.mean(value)) for key, value in batch_statistics.items()])
        #             pbar_train.update(1)
        #             pbar_train.set_description(string_description)
        #
        #     epoch_stats = OrderedDict({})
        #     epoch_train_time = time.time() - epoch_start_time
        #     epoch_stats['current_epoch'] = current_epoch
        #     for k, v in batch_statistics.items():
        #         epoch_stats[k] = np.around(np.mean(v), decimals=4)
        #     epoch_stats['epoch_train_time'] = epoch_train_time
        #
        #     attack.model = self  # updating model of attack!
        #     if x_min_batch_adv is not None:
        #         advs_images_dict[current_epoch] = x_min_batch_adv.detach().clone().cpu().numpy()
        #     else:
        #         advs_images_dict[current_epoch] = None
        #
        #     train_statistics_to_save = epoch_stats
        #     return train_statistics_to_save

        def test_epoch(current_epoch):  # done i think.
            batch_statistics = defaultdict(lambda: [])

            with tqdm(total=len(valid_full)) as pbar_val:
                for i, batch in enumerate(valid_full):
                    x_all, y_all = batch
                    x_all = x_all.to(device=self.device)
                    y_all = y_all.to(device=self.device)

                    output = self.valid_iteration('valid', x_all, y_all,  minority_class)

                    # SAVE BATCH STATS
                    for key, value, in output.items():
                        batch_statistics[key].append(output[key].item())

                    string_description = " ".join(["{}:{:.4f}".format(key, np.mean(value)) for key, value in batch_statistics.items()])
                    pbar_val.update(1)
                    # pbar_val.set_description(string_description)

            with tqdm(total=len(test_full)) as pbar_test:
                for i, batch in enumerate(test_full):
                    x_all, y_all = batch
                    x_all = x_all.to(device=self.device)
                    y_all = y_all.to(device=self.device)

                    output = self.valid_iteration('test', x_all, y_all, minority_class)

                    # SAVE BATCH STATS
                    for key, value, in output.items():
                        batch_statistics[key].append(output[key].item())

                    string_description = " ".join(
                        ["{}: {:.4f}".format(key, np.mean(value)) for key, value in batch_statistics.items()])
                    pbar_test.update(1)
                    # pbar_test.set_description(string_description)

            epoch_stats = OrderedDict({})
            epoch_stats['current_epoch'] = current_epoch
            for k, v in batch_statistics.items():
                epoch_stats[k] = np.around(np.mean(v), decimals=4)

            test_statistics_to_save = epoch_stats
            return test_statistics_to_save

        bpm_overall = defaultdict(lambda: 0)
        bpm_minority = defaultdict(lambda: 0)
        bpm_overall['title'] = 'Best Performing Model Overall'
        bpm_minority['title'] = 'Best Performing Model Minority Class'

        torch.cuda.empty_cache()
        for current_epoch in range(self.num_epochs):
            train_statistics_to_save = train_epoch(current_epoch)
            test_statistics_to_save = test_epoch(current_epoch)

            # save train statistics.
            storage_utils.save_statistics(train_statistics_to_save, file_path=train_results_path)

            # save adversarial images.
            if attack is not None:
                with open(advers_images_path,
                          'wb') as f:  # note you overwrite the file each time but that okay since advs_images_dict grows each epoch.
                    pickle.dump(advs_images_dict, f)

            # save model.
            self.save_model(model_save_dir, model_save_name='model_epoch_{}'.format(str(current_epoch)))
            logger.print(train_statistics_to_save)

            # save bpm statistics
            if test_statistics_to_save['valid_acc'] > bpm_overall['valid_acc']:
                for key, value in test_statistics_to_save.items():
                    bpm_overall[key] = value

            if test_statistics_to_save['valid_acc_class_minority'] > bpm_minority['valid_acc_class_minority']:
                for key, value in test_statistics_to_save.items():
                    bpm_minority[key] = value

            storage_utils.save_statistics(test_statistics_to_save, file_path=valid_and_test_results_path)
            logger.print(test_statistics_to_save)

            if scheduler is not None: scheduler.step()
            for param_group in self.optimizer.param_groups:
                logger.print('learning rate: {}'.format(param_group['lr']))
        return bpm_overall, bpm_minority

    def get_class_stats_helper(self, y_all, y_pred_all, criterion, class_idx):
        # find class specific stats
        y_min = []
        y_pred_min = []
        for i in range(y_all.shape[0]):
            if int(y_all[i].data) == class_idx:
                y_min.append(y_all[i])
                y_pred_min.append(y_pred_all[i])

        loss_min, acc_min = None, None
        if len(y_min) > 0:
            y_min = torch.stack(y_min, dim=0)
            y_pred_min = torch.stack(y_pred_min, dim=0)

            loss_min = (criterion(input=y_pred_min, target=y_min.view(-1)))
            acc_min = self.get_acc_batch(y_min, y_pred_min)
        return loss_min, acc_min

    def get_class_stats(self, type_key, output, minority_class, y_all, y_pred_all, criterion):
        for class_idx in range(10):
            loss_min, acc_min = self.get_class_stats_helper(y_all, y_pred_all, criterion, class_idx)
            if acc_min is None or loss_min is None:
                continue

            if class_idx == minority_class:
                output[type_key + '_acc_class_minority'] = acc_min
                output[type_key + '_loss_class_minority'] = loss_min

            output[type_key + '_acc_class_' + str(class_idx)] = acc_min
            output[type_key + '_loss_class_' + str(class_idx)] = loss_min

    def train_iteration(self, x_all, y_all, minority_class, y_min_map=None, y_min_adv_map=None):
        self.train()
        criterion = nn.CrossEntropyLoss().cuda()

        y_pred_all = self.forward(x_all)
        loss_all = criterion(input=y_pred_all, target=y_all.view(-1))
        self.optimizer.zero_grad()
        loss_all.backward()
        self.optimizer.step()
        acc_all = self.get_acc_batch(y_all, y_pred_all)

        output = {'train_loss': loss_all.data, 'train_acc': acc_all}
        self.get_class_stats('train', output, minority_class, y_all, y_pred_all, criterion)

        # # ADVERSARIAL STATS
        # if y_min_map is not None:
        #     y_adv_min = y_all[y_min_adv_map[0]:y_min_adv_map[1]]
        #     y_pred_adv_min = y_pred_all[y_min_adv_map[0]:y_min_adv_map[1]]
        #
        #     loss_min_adv = criterion(input=y_pred_adv_min, target=y_adv_min.view(-1))
        #     output['loss_adversarial'] = loss_min_adv.data if loss_min_adv is not None else None
        #     output['acc_adversarial'] = self.get_acc_batch(y_adv_min, y_pred_adv_min)

        return output

    def valid_iteration(self, type_key, x_all, y_all, minority_class):
        with torch.no_grad():
            self.eval() # should be eval but something BN - todo: change later if no problems.
            '''
            Evaluating accuracy on whole batch 
            Evaluating accuracy on min examples 
            '''
            criterion = nn.CrossEntropyLoss().cuda()
            y_pred_all = self.forward(x_all)
            loss_all = criterion(input=y_pred_all, target=y_all.view(-1))
            acc_all = self.get_acc_batch(y_all, y_pred_all)
            output = {type_key + '_loss': loss_all.data, type_key + '_acc': acc_all}
            self.get_class_stats(type_key, output, minority_class, y_all, y_pred_all, criterion)
        return output

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
