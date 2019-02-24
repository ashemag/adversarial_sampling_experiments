import torch
import numpy as np

# note: unless stated otherwise all parameter and return types are numpy arrays.

class ModelQuery(object):

    @staticmethod
    def predict(x, model):
        '''
        :param x: batch of images.
            shape: (batch_size, num_channels, height, width)
        :param model: a neural network classifier.
            type: nn.Module
        :return y_sm: softmax probabilities final layer.
            shape: (batch_size, num_classes)
        :return y_pred: predicted class. integer encoded.
            shape: (batch_size,)
        :return y_pred_prob: probability of the predicted class.
            shape: (batch_size,)
        '''

        x = torch.Tensor(x).float()
        y_sm = model(x)
        y_pred_prob, y_pred = torch.max(y_sm.data, 1)

        return y_pred, y_pred_prob, y_sm

    @staticmethod
    def accuracy(x, y, model):
        '''
        :param x: batch of images.
            shape: (batch_size, num_channels, height, width).
        :param y: integere encoded true labels.
            shape: (batch_size,)
        :param model:
            type: nn.Module
        :return acc: accuracy of batch that was passed in.
        '''

        y_pred, _, _ = ModelQuery.predict(x,model)
        y_pred = y_pred.data.numpy()
        acc = np.mean([1 if y_pred[i]==y[i] else 0 for i in range(len(y))])

        return acc


