from mlp_resources import data_providers
from ModelBuilder.base import Network
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import globals
from torchvision.models.densenet import DenseNet

class Temp(DenseNet,Network):
    def __init_(self):
        Network.__init__(self)
        DenseNet.__init__(self)

        print("here")

class SimpleFNN(Network):

    def __init__(self, input_shape, h_out, num_classes, num_channels=3):
        super(SimpleFNN, self).__init__()
        self.input_shape = input_shape
        in_features = input_shape[0]*input_shape[1]  # height*width


        self.hidden_layer = nn.Linear(num_channels * in_features, h_out, bias=False)
        self.output_layer = nn.Linear(h_out, num_classes, bias=False)

    def forward(self, x):
        """
        :param x: tensor.
        :return:
        """
        pred = x
        pred = self.hidden_layer(pred)
        pred = torch.sigmoid(pred)
        pred = self.output_layer(pred)
        # note: output not softmax because CrossEntropyLoss in pytorch does the softmax transform for you
        return pred

def test_mnistdata():
    seed = 9112018;
    rng = np.random.RandomState(seed=seed)
    train_data = data_providers.MNISTDataProvider('train', batch_size=100, rng=rng, max_num_batches=100)

    inputs = train_data.inputs
    targets = train_data.targets

    print("targets: {} inputs: {}".format(inputs.shape,targets.shape))
    pass

def test_load_model():
    '''
    example of how to load a model. model that is loaded is a simple neural network i trained for 50 epochs.
    at each epoch model is saved in folder .../saved_models/test_simple/saved_models_train. model at 40-th
    epoch is loaded.
    '''

    load_model_from = os.path.join(globals.ROOT_DIR,'martins_stuff/SavedModels/SimpleFNN/model_49')
    model = SimpleFNN(input_shape=(28, 28), h_out=100, num_classes=10) # input is mnist data
    model.load_model(load_model_from)
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Test accuracy.
    seed = 9112018; rng = np.random.RandomState(seed=seed)
    train_data = data_providers.MNISTDataProvider('train',batch_size=100, rng=rng,max_num_batches=100) # is an iterator.
    x_batch, y_batch = train_data.next() # arrays size (batch_size,-1)
    acc = model.get_acc_batch(x_batch,y_batch) # calc. accuracy on given batch

    print("accuracy of model on batch: ",acc)

def test_train_and_save():
    '''
    example of training a model and saving its results.
    '''
    rng = np.random.RandomState(seed=9112018)
    train_data = data_providers.MNISTDataProvider('train',batch_size=100, rng=rng,max_num_batches=100)
    model = SimpleFNN(input_shape=(28,28), h_out=100, num_classes=10)
    optimizer = optim.SGD(model.parameters(), lr=1e-1)
    saved_models_dir = os.path.join(globals.ROOT_DIR,'martins_stuff/SavedModels/SimpleFNN')
    train_results_file = os.path.join(globals.ROOT_DIR,'martins_stuff/ExperimentResults/train_results_simple_fnn.txt')
    num_epochs = 50
    model.train_full(train_data, num_epochs, optimizer, train_results_file, saved_models_dir)

def test_evaluating():
    rng = np.random.RandomState(seed=9112018)
    valid_data = data_providers.MNISTDataProvider('valid', batch_size=100, rng=rng, max_num_batches=100)
    model = SimpleFNN(input_shape=(28,28), h_out=100, num_classes=10)
    eval_results_file_path = os.path.join(globals.ROOT_DIR,'martins_stuff/ExperimentResults/eval_results_simple_fnn.txt')
    epochs = [i for i in range(50)]

    model_train_dir = os.path.join(globals.ROOT_DIR,'martins_stuff/SavedModels/SimpleFNN')
    model.evaluate_full(valid_data,epochs,model_train_dir,eval_results_file_path)


def main():
    #test_train_and_save()
    #test_evaluating()
    test_mnistdata()

    pass


if __name__ == '__main__':
    main()