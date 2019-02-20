import matplotlib.pyplot as plt
import numpy as np

class Plotter():
    def __init__(self):
        pass

    @staticmethod
    def multiclass_lineplot(metric_dict):
        plt.figure()

        for class_name in metric_dict.keys():
            xx = metric_dict[class_name]['x_val']
            yy = metric_dict[class_name]['y_val']
            plt.plot(xx, yy, label=metric_dict[class_name]['label'])


        plt.legend()
        plt.show()

    @staticmethod
    def metric_epoch_plot(*metrics):
        '''
        :param metrics: 3-tuple (desired, label_names, filename).
        (1) "desired" is a list of strings e.g. ['acc', ..., 'loss'] corresponding to header names of the file
        you want to retrieve data from.
        (2) "label_names" is also a list of strings ['accuracy',...,'train_loss'] of how you want data on your plot to
        be labelled as.
        (3) "filename" is the path to the file you want to retrieve data from.

        :return:
        line multi-class line plot of performance metrics over epochs. can for e.g. be used to plot a learning curve.
        '''

        for (target_names,labels,filename) in metrics:
            metrics_dict = {label: {} for label in labels}

            with open(filename, 'r') as file:
                source_names = file.__next__().split()
                if bool(set(source_names) & set(target_names)):  # if any items in common
                    tgt2src = {i : source_names.index(target_name) for i,target_name in enumerate(target_names)}
                    for row_idx,line in enumerate(file):
                        row = line.split()
                        for tgt,src in tgt2src.items():
                            target_value = float(row[src])
                            metrics_dict[tgt]['x_val'].append(row_idx)
                            metrics_dict[tgt]['y_val'].append(target_value)
                            metrics_dict[tgt]['label'] = labels[tgt]

        Plotter.multiclass_lineplot(metrics_dict)