import matplotlib.pyplot as plt
import numpy as np

class ImageDataViewer():
    def __init__(self,data):
        self.inputs = data[0]
        self.int_targets = data[1]

        pass

    def class_member(self, grid_shape, target):
        '''
        :param grid_shape:
        :param target: is an integer.
        '''

        data = (self.inputs,self.int_targets)
        subset = DataHandler.subset(data, targets=[target],shuffle=True)

        num_images = grid_shape[0]+grid_shape[1]
        images = [subset[0][i] for i in range(num_images)]
        labels = [subset[0][i] for i in range(num_images)]


        pass

    def multi_image_plot(self,data,grid_shape):
        images, labels = data[0], data[1]

        for i, img in enumerate(images):
            plt.subplot(grid_shape[0], grid_shape[1], i + 1)
            plt.imshow(img)

            # label which class they belong to.

        plt.show()





class DataHandler():
    DEFAULT_SEED = 20112018

    @staticmethod
    def subset(data, targets,shuffle=False,rng=None):
        images = data[0]
        labels = data[1]
        out = [(images[i],labels[i]) for i in range(len(images)) if labels[i] in targets]
        if shuffle:
            out = DataHandler.shuffle(out,rng)
        return out

    @staticmethod
    def shuffle(data,rng=None):
        if rng is None:
            rng = np.random.RandomState(DataHandler.DEFAULT_SEED)
        images, labels = data[0], data[1]
        perm = rng.permutation(len(images))
        shuffled = (images[perm], labels[perm])
        return shuffled


def bar_chart_subplots(data_dict_list):
    # argv must be data_dicts of form data_dict[x_val] = count of x_val.
    import matplotlib.pyplot as plt

    num_plots = len(data_dict_list)
    for i,data_dict in enumerate(data_dict_list):
        counts = [y for y in data_dict.values()]
        x_vals = [x for x in data_dict.keys()]

        plt.subplot(1,num_plots,i+1)
        plt.bar(x_vals, height=counts, align='center', alpha=0.5)  # align is position of bar relative to x-ticks
        plt.xticks(x_vals)
        plt.xlabel('Sentence length')
        plt.ylabel('Count')

    plt.show()


def cifar10_test():
    viewer = ImageDataViewer(cifar10)
    viewer.class_member(grid_size=(4,4),int_label=8,rng=True)

    pass