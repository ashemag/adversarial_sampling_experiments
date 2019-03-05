import os
import torch.optim as optim
import torch

from attacks.advers_attacks import LInfProjectedGradientAttack
from data_io import ImageDataIO
from data_providers import DataProvider
from globals import ROOT_DIR
from models.densenet import DenseNet121

'''
1. initialize densenet121 for cifar10.
'''

model = DenseNet121()

'''
STEP 2: initialize the adversarial attack that will be used during adversarial training. 
'''

attack = LInfProjectedGradientAttack(
            model=model,
            steps=40, alpha=0.01, epsilon=0.3, rand=True, targeted=False
        )

'''
STEP 3: specify what the minority class in the dataset is. minority class data is the data that we
will be adversarially attacking.
'''

labels_minority = [0] # must be a list.

'''
STEP 4: create a data provider for the training and validation set. the code below uses MNIST data.

note: ImageDataIO is a convenience class for loading specific data into memory into the following format:
x numpy array, shape (batch_size, num_channels, height, width), y numpy array, shape (batch_size,).
'''

x, y = ImageDataIO.cifar10('train') # convenience class to extract load mnist-train.npz into memory.
x_val, y_val = ImageDataIO.cifar10('valid')
dp_train = DataProvider(x, y, batch_size=100, max_num_batches=10, make_one_hot=False, rng=None)
dp_valid = DataProvider(x, y, batch_size=100, max_num_batches=10, make_one_hot=False, rng=None)

'''
STEP 5: call the "advers_train_and_evaluate" function that "model" has. this function takes a bunch of parameters.
most arguments are familiar. the one that is different is "advs_images_file". this is the pickle file that adversarial
images are saved to every epoch. recall how Antreas said it would be nice to see the progression of adversarial images
created after each epoch of training our model. this is the file that saves those images.

the contents of "advs_images_file" can be extrated using pickle. the content is a dictionary of the form:

dict[epoch_num] = x,  where x is a numpy array of shape (num_channels, height, width) that corr. to an image.
'''

num_epochs = 5
BATCH_SIZE = 64
LEARNING_RATE = .1
WEIGHT_DECAY = 1e-4
MOMENTUM = .9

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                                    momentum=MOMENTUM,
                                    nesterov=True,
                                    weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0001)

model.advers_train_and_evaluate(
    labels_minority=labels_minority,
    attack = attack,
    advs_images_file=os.path.join(ROOT_DIR,'ExperimentResults/advers_images_cifar10.pickle'),
    m_batch_size=6,
    o_batch_size=64,
    num_epochs=num_epochs,
    optimizer=optimizer,
    model_save_dir=os.path.join(ROOT_DIR,'saved_models/cifar10_advers_model'),
    train=(dp_train, 'ExperimentResults/cifar10_advers_train_results.txt'),
    scheduler=scheduler,
    valid = (dp_valid,'ExperimentResults/cifar10_advers_valid_results.txt')
)

