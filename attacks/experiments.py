import numpy as np
import os
from tqdm import tqdm

from adversarial_sampling_experiments.attacks.advers_attacks import *


def test_attacks(model,x_obs,y_true):
    '''
    compares original vs. augmented image
    :return:
    '''
    # to generate trippy images show what attacks look like on advers trained network

    fast_attack = FastGradientSignAttack(model,alpha=0.1)
    pgd_attack = LInfProjectedGradientAttack(model,steps=100,alpha=0.1/100,epsilon=1/2,rand=True)
    pgd_attack_penalty =  \
        LInfProjectedGradientAttackPenalty(model,steps=100,alpha=0.1/100,epsilon=1/2,gamma=100,
                                           rand=True,targeted=False)

    # __init__(self,model,steps,alpha,epsilon,gamma,rand=False,targeted=False):

    x_adv_fast = fast_attack(x_obs,y_true)
    x_adv_pgd = pgd_attack(x_obs,y_true)
    x_adv_pgd_penalty = pgd_attack_penalty(x_obs,y_true)

    # plot images

    images_dict = {'original':x_obs, 'advers_fast':x_adv_fast, 'advers_pgd':x_adv_pgd,
                   'advers_pgd_penalty': x_adv_pgd_penalty}
    num_subplots = len(images_dict.keys())
    plt.figure()

    for i, (name, img) in enumerate(images_dict.items()):
        plt.subplot(1, num_subplots, i + 1)
        plt.imshow(img.reshape((28, 28)), cmap='Greens')
        plt.xlabel(name)

    plt.subplots_adjust(hspace=0.5)
    plt.show()



def visualize_input_space_loss_gradients():
    # replicates figure 2 from https://arxiv.org/pdf/1805.12152.pdf
    pass

if __name__ == '__main__':

    pass

