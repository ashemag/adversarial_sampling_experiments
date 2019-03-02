from adversarial_sampling_experiments.data_subsetter import DataSubsetter
from adversarial_sampling_experiments.data_providers import DataProvider
from adversarial_sampling_experiments.attacks.advers_attacks import LInfProjectedGradientAttack
import pickle
import os
from adversarial_sampling_experiments.globals import ROOT_DIR
import torch.optim as optim
from adversarial_sampling_experiments.models.simple_fnn import FeedForwardNetwork

'''
1. load in an empty model with settings that match the data.
'''

model = FeedForwardNetwork(img_shape=(1, 28, 28), num_classes=10) # 1.

labels_minority = [0] # must be a list
attack = LInfProjectedGradientAttack(
            model=model,
            steps=40, alpha=0.01, epsilon=0.3, rand=True, targeted=False
        )

x, y = ImageDataIO.mnist('train')
x_val, y_val = ImageDataIO.mnist('valid')
dp_train = DataProvider(x, y, batch_size=100, max_num_batches=10, make_one_hot=False, rng=None)
dp_valid = DataProvider(x, y, batch_size=100, max_num_batches=10, make_one_hot=False, rng=None)

model.advers_train_and_evaluate(
    labels_minority=labels_minority,
    attack = attack,
    advs_images_file=os.path.join(ROOT_DIR,'ExperimentResults/advers_images.pickle'),
    num_epochs=2,
    optimizer=optim.SGD(model.parameters(), lr=1e-1),
    model_save_dir=os.path.join(ROOT_DIR,'saved_models/simple_advers_model'),
    train=(dp_train, 'ExperimentResults/simple_advers_train_results.txt'),
    scheduler=None,
    valid = (dp_valid,'ExperimentResults/simple_advers_valid_results.txt')