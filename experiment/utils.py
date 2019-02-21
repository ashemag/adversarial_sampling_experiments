import numpy as np
import torch
from collections import OrderedDict
import os


def test():
    from models.simple_fnn import SimpleFNN
    x = np.random.normal(0,1,size=(5,24*24*1))
    print("x shape: ",x.shape)

    model = SimpleFNN((24,24),h_out=100,num_classes=10,num_channels=1)

    confidence,pred = ModelMetrics.confidence(model,x)

    print("type: ",type(confidence),"a shape: ",confidence.shape, "b shape: ",pred.shape)

    print(confidence)
    print("pred: ",pred)
    pass

class ModelMetrics(object):

    @staticmethod
    def confidence(model,x):
        '''
        calculates for each observation in a batch how confident the model is in it's classification.

        :param model: nn.Module
        :param x: array. shape: (batch_size, -1) e.g. rgb image (batch_size,3,32,32)
        :return: probability of predicted label (i.e. confidence), and (integer encoded) predicted label.
        '''

        x = torch.Tensor(x).float()
        y_pred = model(x) # shape: (num_batches, num_classes)
        y_conf_best, y_pred_best = torch.max(y_pred.data,1) # max, argmax. shape: (-1,)

        return y_conf_best.data.numpy().reshape(-1,1), y_pred_best.data.numpy().reshape(-1,1)


def get_acc_batch(model ,x_batch ,y_batch ,y_batch_pred=None):
    """
    :param x_batch: array or tensor
    :param y_batch: array, one-hot-encoded
    :param y_batch_pred:  tensor, (because results from model forward pass)
    """

    if type(x_batch) is np.ndarray:
        x_batch = torch.Tensor(x_batch).float()

    if y_batch_pred is None:
        y_batch_pred = model(x_batch)

    y_batch_int = np.argmax(y_batch ,axis=1)
    y_batch_int = torch.Tensor(y_batch_int).long()
    _, y_pred_batch_int = torch.max(y_batch_pred.data, 1)  # argmax of predictions
    acc = np.mean(list(y_pred_batch_int.eq(y_batch_int.data).cpu()))  # compute accuracy

    return acc

def save_statistics(statistics_to_save,file_path):
    '''
    :param statistics_to_save: dict, val type is float
    :param file_path: e.g. file_path = "C:/test_storage_utils/dir2/test.txt"
    '''
    if type(statistics_to_save) is not OrderedDict:
        raise TypeError('statistics_to_save must be OrderedDict instead got {}'.format(type(statistics_to_save)))

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path,'a+') as f: # append mode + creates if doesn't exist
        header = ""
        line = ""
        for i,key in enumerate(statistics_to_save.keys()):
            val = statistics_to_save[key]
            if i==0:
                line = line + "{:.4f}".format(val)
                header = header + key
            else:
                line = line + "\t" + "{:.4f}".format(val)
                header = header + "\t" + key
        if os.stat(file_path).st_size == 0:  # if empty
            f.write(header+"\n")
        f.write(line+"\n")

def save_model(model, model_save_dir, model_save_name):
    state = dict()
    state['network'] = model.state_dict()  # save network parameter and other variables.
    model_path = os.path.join(model_save_dir, model_save_name)

    directory = os.path.dirname(model_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(state, f=model_path)

def load_model(model, model_path):
    state = torch.load(f=model_path)
    model.load_state_dict(state_dict=state['network'])
    return model

if __name__ == '__main__':
    test()


    pass