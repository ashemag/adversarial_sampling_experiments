from adversarial_sampling_experiments import data_providers
from adversarial_sampling_experiments import globals

from adversarial_sampling_experiments.models.base import Network
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from adversarial_sampling_experiments.attacks.data_augmenter import *
from adversarial_sampling_experiments.data_providers import *

class FeedForwardNetwork(Network):

    def __init__(self, img_shape, num_classes, config_list=None):
        '''
        :param img_shape:
            type: tuple.
            format: (num_channels, height, width).
        '''
        super(FeedForwardNetwork, self).__init__()
        self.img_num_channels, self.img_height, self.img_width = img_shape

        if config_list is None:
            self.config_list = [
                {'type': 'fc', 'out_features': 100, 'bias': False,'nl':'relu','dropout':None}
            ]
            classifier_pattern = [
                {'type': 'fc', 'out_features': num_classes, 'bias': False,'nl':None,'dropout':None}
            ]

            self.config_list += classifier_pattern
        else:
            self.config_list = config_list

        for i, config_dict in enumerate(self.config_list):
            if config_dict['type'] == 'fc':
                if 'bias' not in config_dict.keys():
                    config_dict['bias'] = False
                if 'repeat' not in config_dict.keys():
                    config_dict['repeat'] = 1
                if 'bn' not in config_dict.keys():
                    config_dict['bn'] = False
                if 'nl' not in config_dict.keys():
                    config_dict['nl'] = None

        self._config_keys = {
            'stride': 's',
            'kernel_size': 'k',
            'padding': 'p',
            'out_channels': 'd',
            'out_features': 'out_features',
            'bias': 'bias'
        }

        self.layer_dict = nn.ModuleDict()
        self.build_module()

    def build_module(self):
        def add_fc_layer(out,config_dict,fc_idx):
            if len(out.shape) > 2:
                out = out.view(out.shape[0], -1)  # flatten into (batch_size, -1)

            label = ''
            modules = []
            if config_dict['dropout'] is not None:
                modules.append(nn.Dropout(config_dict['dropout'],inplace=False))
                label += 'dropout'

            fc =  nn.Linear(
                in_features=out.shape[1],
                out_features=config_dict[self._config_keys['out_features']],
                bias=config_dict[self._config_keys['bias']]
            )
            modules.append(fc)
            label += '-fc'

            if config_dict['nl'] is not None and config_dict['nl']=='relu':
                modules.append(nn.ReLU(inplace=True))
                label += '-relu'

            self.layer_dict['fc_{}'.format(fc_idx)] = nn.Sequential(*modules)

            # update the depth of the current volume (used for creating subsequent layers)
            out = self.layer_dict['fc_{}'.format(fc_idx)](out)
            print(out.shape, label[1:]) # e.g. "-fc-relu" becomes "fc-relu"

            # update next idx of fc layer (used for naming the layers)
            fc_idx += 1

            return out, fc_idx

        print("building feed-forward network module")
        x = torch.zeros(
            (2, self.img_num_channels, self.img_height, self.img_width))  # dummy batch to infer layer shapes.
        out = x
        print(out.shape, "input")

        fc_idx = 0
        for layer_config_dict in self.config_list:
            if layer_config_dict['type'] == 'fc':
                out, fc_idx = add_fc_layer(out, layer_config_dict, fc_idx)

    def forward(self, x):
        pred = x
        for k in self.layer_dict.keys():  # dict is ordered
            if k[0:2] == 'fc':  # can be e.g. fc_2
                pred = pred.view(pred.shape[0], -1)  # flatten
            pred = self.layer_dict[k](pred)

        return pred


def test_load_model():
    '''
    example of how to load a model. model that is loaded is a simple neural network i trained for 50 epochs.
    at each epoch model is saved in folder .../saved_models/test_simple/saved_models_train. model at 40-th
    epoch is loaded.
    '''

    load_model_from = os.path.join(globals.ROOT_DIR,'martins_stuff/SavedModels/SimpleFNN/model_49')
    model = FeedForwardNetwork(img_shape=(28, 28), h_out=100, num_classes=10) # input is mnist data
    model.load_model(load_model_from)
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Test accuracy.
    seed = 9112018; rng = np.random.RandomState(seed=seed)
    train_data = data_providers.MNISTDataProvider('train',batch_size=100, rng=rng,max_num_batches=100) # is an iterator.
    x_batch, y_batch = train_data.next() # arrays size (batch_size,-1)
    acc = model.get_acc_batch(x_batch,y_batch) # calc. accuracy on given batch

    print("accuracy of model on batch: ",acc)

def test_train_and_save(saved_models_dir,train_results_file):
    model = FeedForwardNetwork(img_shape=(1, 28, 28), num_classes=10)

    x, y = DataAugmenter.load_data(
        filename=os.path.join(globals.ROOT_DIR, 'data/mnist-train.npz')
    )

    rng = np.random.RandomState(seed=9112018)
    train_data = DataProvider(x, y, batch_size=100, max_num_batches=100, make_one_hot=False, rng=rng)

    model.train_full(
        train_data=train_data,
        num_epochs=50,
        optimizer=optim.SGD(model.parameters(), lr=1e-1),
        train_file_path=train_results_file,
        model_save_dir=saved_models_dir,
        integer_encoded = True
    )

# def test_train_and_save(saved_models_dir,train_results_file):
#     '''
#     example of training a model and saving its results.
#     '''
#     rng = np.random.RandomState(seed=9112018)
#     train_data = data_providers.MNISTDataProvider('train',batch_size=100, rng=rng,max_num_batches=100)
#     model = FeedForwardNetwork(img_shape=(28, 28), num_channels=1, h_out=100, num_classes=10)
#     optimizer = optim.SGD(model.parameters(), lr=1e-1)
#     # saved_models_dir = os.path.join(globals.ROOT_DIR,'martins_stuff/SavedModels/SimpleFNN')
#     # train_results_file = os.path.join(globals.ROOT_DIR,'martins_stuff/ExperimentResults/train_results_simple_fnn.txt')
#     num_epochs = 50
#     model.train_full(train_data, num_epochs, optimizer, train_results_file, saved_models_dir)

def test_evaluating(model_train_dir,eval_results_file_path):
    rng = np.random.RandomState(seed=9112018)
    valid_data = data_providers.MNISTDataProvider('valid', batch_size=100, rng=rng, max_num_batches=100)
    model = FeedForwardNetwork(img_shape=(28, 28), num_channels=1, h_out=100, num_classes=10)
    # eval_results_file_path = os.path.join(globals.ROOT_DIR,'martins_stuff/ExperimentResults/eval_results_simple_fnn.txt')
    epochs = [i for i in range(50)]

    # model_train_dir = os.path.join(globals.ROOT_DIR,'martins_stuff/SavedModels/SimpleFNN')
    model.evaluate_full(valid_data,epochs,model_train_dir,eval_results_file_path)


def main():
    saved_models_dir = os.path.join(globals.ROOT_DIR, 'saved_models/simple_fnn')
    train_results_file_path = os.path.join(globals.ROOT_DIR,'ExperimentResults/simple_fnn/train_results.txt')
    test_train_and_save(saved_models_dir,train_results_file_path)

    # test_evaluating()

    pass


if __name__ == '__main__':
    main()