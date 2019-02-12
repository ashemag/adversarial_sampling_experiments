import numpy as np
import torch
import torch.nn.functional as F
import time
import os
from ModelBuilder import storage_utils
from tqdm import tqdm
import sys
from collections import OrderedDict


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.num_epochs = None
        self.train_data = None
        self.optimizer = None
        self.train_file_path = None
        self.cross_entropy = None

        use_gpu = False

        if torch.cuda.is_available() and use_gpu:  # checks whether a cuda gpu is available and whether the gpu flag is True
            self.device = torch.device('cuda')  # sets device to be cuda
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # sets the main GPU to be the one at index 0 (on multi gpu machines you can choose which one you want to use by using the relevant GPU ID)
            print("use GPU")
        else:
            print("use CPU")
            self.device = torch.device('cpu')  # sets the device to be CPU

    def get_acc_batch(self,x_batch,y_batch,y_batch_pred=None):
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

        y_batch_int = np.argmax(y_batch,axis=1)
        y_batch_int = torch.Tensor(y_batch_int).long().to(device=self.device)
        _, y_pred_batch_int = torch.max(y_batch_pred.data, 1)  # argmax of predictions
        acc = np.mean(list(y_pred_batch_int.eq(y_batch_int.data).cpu()))  # compute accuracy

        return acc

    def train_iter(self,x_train_batch,y_train_batch):
        """
        :param x_train_batch: array, one-hot-encoded
        :param y_train_batch: array, one-hot-encoded
        :return:
        """

        # CrossEntropyLoss. Input: (N,C), target: (N) each value is integer encoded.

        self.train()
        y_train_batch_int = np.argmax(y_train_batch,axis=1)
        y_train_batch_int = torch.Tensor(y_train_batch_int).long().to(device=self.device)
        x_train_batch = torch.Tensor(x_train_batch).float().to(device=self.device)
        y_pred_batch = self(x_train_batch) # model forward pass
        loss = F.cross_entropy(input=y_pred_batch,target=y_train_batch_int) # self.cross_entropy(input=y_pred_batch,target=y_train_batch_int)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        acc_batch = self.get_acc_batch(x_train_batch,y_train_batch,y_pred_batch)

        return loss.data, acc_batch

    def save_train_epoch_results(self, batch_statistics,train_file_path):
        statistics_to_save = {"train_acc":0, "train_loss":0, "epoch_train_time":0,"current_epoch":0}
        statistics_to_save["current_epoch"] = batch_statistics["current_epoch"]
        statistics_to_save["epoch_train_time"] = batch_statistics["epoch_train_time"]

        for key, value in batch_statistics.items():
            if key not in ["current_epoch","epoch_train_time"]:
                batch_values = np.array(batch_statistics[key])
                epoch_val = np.mean(batch_values)  # get mean of all metrics of current epoch metrics dict
                statistics_to_save[key] = np.around(epoch_val,decimals=4)

        print(statistics_to_save)
        storage_utils.save_statistics(statistics_to_save,train_file_path)

    def train_full(self, train_data, num_epochs, optimizer,train_file_path,model_save_dir):
        self.num_epochs = num_epochs
        self.train_data = train_data
        self.optimizer = optimizer
        self.train_file_path = train_file_path
        self.cross_entropy = torch.nn.CrossEntropyLoss()

        for current_epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            batch_statistics = {"train_acc": [], "train_loss": []}

            for i, (x_train_batch, y_train_batch) in tqdm(enumerate(self.train_data), file=sys.stdout):  # get data batches
                loss_batch, accuracy_batch = self.train_iter(x_train_batch, y_train_batch)  # take a training iter step
                batch_statistics["train_loss"].append(loss_batch)  # add current iter loss to the train loss list
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

    def train_iter(self,x_train_batch,y_train_batch):
        """
        :param x_train_batch: array
        :param y_train_batch: array, one-hot-encoded
        :return:
        """

        # CrossEntropyLoss. Input: (N,C), target: (N) each value is integer encoded.

        self.train()
        y_train_batch_int = np.argmax(y_train_batch, axis=1)
        y_train_batch_int = torch.Tensor(y_train_batch_int).long().to(device=self.device)
        x_train_batch = torch.Tensor(x_train_batch).float().to(device=self.device)
        y_pred_batch = self.forward(x_train_batch) # model forward pass
        loss = F.cross_entropy(input=y_pred_batch,target=y_train_batch_int) # self.cross_entropy(input=y_pred_batch,target=y_train_batch_int)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        acc_batch = self.get_acc_batch(x_train_batch,y_train_batch,y_pred_batch)

        return loss.data, acc_batch


    def run_evaluation_iter(self,x_batch,y_batch):
        '''
        :param x_batch:
        :param y_batch:
        :return:
        '''
        self.eval()
        y_batch_int = np.argmax(y_batch, axis=1)
        y_batch_int_tens = torch.Tensor(y_batch_int).long().to(device=self.device)
        x_batch_tens = torch.Tensor(x_batch).float().to(device=self.device)
        y_batch_pred_tens = self(x_batch_tens)  # model forward pass
        loss_batch = F.cross_entropy(input=y_batch_pred_tens,target=y_batch_int_tens)
        acc_batch = self.get_acc_batch(x_batch_tens,y_batch,y_batch_pred_tens)

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
                batch_statistics["train_loss"].append(loss_batch)
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

def main():
    # test experiment builder
    pass

if __name__ == '__main__':
    main()