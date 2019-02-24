import numpy as np

class DataSubsetter():
    DEFAULT_SEED = 20112018

    @staticmethod
    def condition_on_label(x, y, labels, shuffle=False, rng=None):
        x_sub = np.array([x[i] for i in range(len(x)) if y[i] in labels])
        y_sub = np.array([y[i] for i in range(len(x)) if y[i] in labels])
        if shuffle:
            x_sub, y_sub = DataSubsetter.shuffle(x_sub, y_sub, rng)

        return x_sub, y_sub

    @staticmethod
    def shuffle(x, y,rng=None):
        if rng is None:
            rng = np.random.RandomState(DataSubsetter.DEFAULT_SEED)
        images, labels = x, y
        perm = rng.permutation(len(images))

        return images[perm], labels[perm]

    # @staticmethod
    # def better_than(model,data,lb,metric='accuracy'):
    #     x, y = data
    #     probs, y_pred = ModelMetrics.confidence(model,x)
    #
    #     x_out, y_out = [],[]
    #     probs_out, y_pred_out = [],[]
    #     for i in range(len(x)):
    #         if probs[i] >= lb:
    #             x_out.append(x[i])
    #             y_out.append(y[i])
    #             y_pred_out.append(y_pred[i])
    #             probs_out.append(probs[i])
    #
    #     x_out, y_out = np.array(x_out), np.array(y_out)
    #     data_out = (x_out,y_out)
    #
    #     return data_out, probs_out, y_pred_out

    @staticmethod
    def worse_than(model,ub,metric='accuracy'):

        pass

    @staticmethod
    def between(model,lb,ub,metric='accuracy'):

        pass