import numpy as np
import torch
from collections import OrderedDict
import os

def get_acc_batch(self ,x_batch ,y_batch ,y_batch_pred=None):
    """
    :param x_batch: array or tensor
    :param y_batch: array, one-hot-encoded
    :param y_batch_pred:  tensor, (because results from model forward pass)
    """

    if type(x_batch) is np.ndarray:
        x_batch = torch.Tensor(x_batch).float().to(device=self.device)

    if y_batch_pred is None:
        y_batch_pred = self(x_batch)

    y_batch_int = np.argmax(y_batch ,axis=1)
    y_batch_int = torch.Tensor(y_batch_int).long().to(device=self.device)
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