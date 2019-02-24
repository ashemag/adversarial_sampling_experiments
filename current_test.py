from adversarial_sampling_experiments.data_providers import ImageDataGetter
from adversarial_sampling_experiments.data_providers import DataProvider
from adversarial_sampling_experiments.models.simple_fnn import FeedForwardNetwork
from adversarial_sampling_experiments.attacks.advers_attacks import LInfProjectedGradientAttack
from adversarial_sampling_experiments.attacks.data_augmenter import DataAugmenter
from adversarial_sampling_experiments.data_viewer import ImageDataViewer
from adversarial_sampling_experiments.model_queries import ModelQuery

import os
from matplotlib import pyplot as plt
from adversarial_sampling_experiments.globals import ROOT_DIR
import numpy as np


def show_images(x,model):
    plot_dict = {}
    fig, axs = plt.subplots(4, 4)
    axs = np.reshape(axs, (-1,))

    pred, _, _ = ModelQuery.predict(x, model)

    labels = ['true: {} \n pred: {}'.format(y[i], pred[i]) for i in range(len(y))]

    # labels = [i for i in range(len(y))]

    for i, (ax, img, label) in enumerate(zip(axs, x, labels)):
        plot_dict[i] = {'ax': ax, 'img': img, 'x_label': label}

    ImageDataViewer.grid(plot_dict, hspace=0.75)

x, y = ImageDataGetter.mnist(filename=os.path.join(ROOT_DIR, 'data/mnist-train.npz')) # (1)
data_iterator = DataProvider(x,y,batch_size=16,max_num_batches=1,make_one_hot=False,rng=None) # (2)
x, y = data_iterator.__next__()

model = FeedForwardNetwork(img_shape=(1, 28, 28), num_classes=10)
model.load_model(
    model_path=os.path.join(ROOT_DIR,'saved_models/simple_fnn/model_epoch_49') # (3)
)

attack = LInfProjectedGradientAttack(
    model = model,
    steps = 40,
    alpha = 0.01, # step size
    epsilon = 0.3,
    rand = True, # initialize at uniformly random feasible point
    targeted=False
)

x_adv = DataAugmenter.advers_attack(x,y,attack=attack)

show_images(x_adv,model)
show_images(x,model)







