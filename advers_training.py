from data_providers import DataProvider
from attacks.advers_attacks import LInfProjectedGradientAttack
import os
from globals import ROOT_DIR
import torch.optim as optim
from models.simple_fnn import FeedForwardNetwork
from data_io import ImageDataIO

'''
STEP 1: initialize a model that is suitable for the data you will be training it on. in case of MNIST
img_shape = (1,28,28). in case of CIFAR img_shape = (2,32,32).
'''

model = FeedForwardNetwork(img_shape=(1, 28, 28), num_classes=10) # 1.

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

x, y = ImageDataIO.mnist('train') # convenience class to extract load mnist-train.npz into memory.
x_val, y_val = ImageDataIO.mnist('valid')
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

model.advers_train_and_evaluate(
    labels_minority=labels_minority,
    attack = attack,
    advs_images_file=os.path.join(ROOT_DIR,'ExperimentResults/advers_images.pickle'),
    m_batch_size = 10,
    o_batch_size = 100,
    num_epochs=2,
    optimizer=optim.SGD(model.parameters(), lr=1e-1),
    model_save_dir=os.path.join(ROOT_DIR,'saved_models/simple_advers_model'),
    train=(dp_train, 'ExperimentResults/simple_advers_train_results.txt'),
    scheduler=None,
    valid = (dp_valid,'ExperimentResults/simple_advers_valid_results.txt')
)

