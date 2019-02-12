import numpy as np
import os
from tqdm import tqdm
from martins_stuff.advers_attacks import *
from martins_stuff.ModelBuilder.simple_fnn import SimpleFNN

def prep_data_for_advers_train(x, y_onehot, attack, save_aug_data=False, save_path_npzfile=None):
    '''
    :param x: numpy array shape (batch_size, img_height, img_width)
    :param y_onehot: numpy array shape (batch_size, num_classes)
    :param attack: attack object e.g. LInfProjectedGradientAttack in advers_attacks.py

    advers training essentially amounts to training on the perturbed images.
    this function prepares the data for advers training. you can choose
    to save this data.

    :return: augmented data for advers training
    '''
    if save_aug_data:
        if save_path_npzfile is None: raise Exception('you have to specify dir path to save data to')

    x_adv = np.zeros_like(x)

    for i in tqdm(range(len(x))):
        x_adv[i] = attack(x[i])

    if save_aug_data:
        directory = os.path.dirname(save_path_npzfile)
        if not os.path.exists(directory):
            os.makedirs(directory)
        data_to_save = {'x_adv':x_adv,'y_onehot':y_onehot}
        np.savez(save_path_npzfile, data_to_save)

    return x_adv, y_onehot


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

def load_test_model(pretrained=False):
    if pretrained:
        from globals import ROOT_DIR
        model_path = os.path.join(ROOT_DIR, 'martins_stuff/SavedModels/SimpleFNN/model_49')
        model = SimpleFNN(input_shape=(28,28), h_out=100, num_classes=10) # (1)
        model.load_model(model_path)
        '''
        Remarks:
        (1) in order to load model you need to create a model that has same structure as the
        one you saved.
        '''
    return model

if __name__ == '__main__':
    model = load_test_model(pretrained=True)

    seed = 9112018
    rng = np.random.RandomState(seed=seed)
    data = data_providers.MNISTDataProvider('train', batch_size=1, rng=rng, max_num_batches=100)
    x_obs, y_true = data.next()

    test_attacks(model,x_obs,y_true)

