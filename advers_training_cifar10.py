import os
import torch.optim as optim
import torch

from attacks.advers_attacks import LInfProjectedGradientAttack
from data_io import ImageDataIO
from data_providers import DataProvider
from globals import ROOT_DIR
from models.densenet import DenseNet121
import sys
from data_subsetter import DataSubsetter
from models.base import Logger

logger = Logger(stream=sys.stderr, disable=False)

class AdversExperiment():
    def __init__(self,which_model,use_gpu=True):
        self.labels_minority = [8]  # must be a list.
        self.labels_majority = [i for i in range(10) if i not in [8]]
        self.num_epochs = 120
        BATCH_SIZE = 64
        LEARNING_RATE = .1
        WEIGHT_DECAY = 1e-4
        MOMENTUM = .9

        self.model = AdversExperiment.get_model(which_model=which_model) # densenet for cifar10
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=LEARNING_RATE,momentum=MOMENTUM,nesterov=True,weight_decay=WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs, eta_min=0.0001)
        if use_gpu: self.model.use_gpu(gpu_ids='0')  # must come before defining the attack (so that attack uses GPU as well).

    @staticmethod
    def get_model(which_model):
        model = None
        if which_model=='densenet':
            model = DenseNet121()
        if which_model=='mnist_simple':
            from models.simple_fnn import FeedForwardNetwork
            config_list = [{'type': 'fc', 'out_features': 100, 'bias': False, 'nl': 'relu', 'dropout': None}]
            model = FeedForwardNetwork(img_shape=(1, 28, 28), num_classes=10, config_list=config_list)
        return model

    def init_dataproviders(self, majority_batch_size, minority_batch_size, minority_percentage,which_data):
        '''
        valid dataprovider (for whole test) (done)
        train_majority_dataprovider (done)
        valid_minority_dataprovider (for target_valid_acc)
        train_minority_dataprovider (for target_train_acc + adversarial images.) (done)
        '''
        if which_data=='cifar10':
            x_train, y_train = ImageDataIO.cifar10('train')
            x_test, y_test = ImageDataIO.cifar10('test')
        elif which_data=='mnist':
            x_train, y_train = ImageDataIO.mnist('train')
            x_test, y_test = ImageDataIO.mnist('test')
        else:
            return

        x_train_majority, y_train_majority = \
            DataSubsetter.condition_on_label(x_train, y_train,labels=self.labels_majority,shuffle=True,rng=None)

        self.train_majority_dataprovider = DataProvider(
            inputs = x_train_majority,
            targets = y_train_majority,
            batch_size=majority_batch_size,
            max_num_batches=-1,
            make_one_hot=False,
            rng=None,
            with_replacement=False
        )

        x_train_minority, y_train_minority = \
            DataSubsetter.condition_on_label(x_train, y_train, labels=self.labels_minority, shuffle=True, rng=None)
        size_minority = int(len(x_train_minority) * minority_percentage)
        x_train_minority = x_train_minority[:size_minority]

        self.train_minority_dataprovider = DataProvider(
            inputs = x_train_minority,
            targets = y_train_minority,
            batch_size = minority_batch_size,
            max_num_batches=-1,
            make_one_hot=False,
            rng=None,
            with_replacement=True
        )

        self.valid_dataprovider = DataProvider(
            inputs = x_test,
            targets = y_test,
            batch_size=100, # doesn't matter (since only used for validation)
            max_num_batches=-1,
            make_one_hot=False,
            rng=None,
            with_replacement=False
        )

        x_test_minority, y_test_minority = \
            DataSubsetter.condition_on_label(x_test, y_test, labels=self.labels_minority, shuffle=True, rng=None)

        self.valid_minority_dataprovider = DataProvider(
            inputs = x_test_minority,
            targets = y_test_minority,
            batch_size=100,  # doesn't matter (since only used for validation)
            max_num_batches=-1,
            make_one_hot=False,
            rng=None,
            with_replacement=False
        )
        logger.print('################ data information ################')
        logger.print('train size: {}'.format(len(x_train)))
        logger.print('train minority size: {}. percentage: {}'.format(len(x_train_minority),minority_percentage))
        logger.print('train majority size: {}'.format(len(x_train_majority)))
        logger.print('test size: {}'.format(len(x_test)))
        logger.print('test minority size: {}'.format(len(x_test_minority)))
        logger.print('##################################################')

    def mnist_test_exp(self,results_dir):
        epsilon_attack = 4 / 255  # pixels in mnist data are between 0 and 1 i think (verify)

        logger.print('creating attack.')
        attack = LInfProjectedGradientAttack(
            model=self.model,
            steps=1, alpha=0.01, epsilon=epsilon_attack, rand=True, targeted=False
        )

        self.init_dataproviders(
            majority_batch_size=64,
            minority_batch_size=6,
            minority_percentage=1,
            which_data='mnist'
        )

        self.model.advers_train_and_evaluate(
            train_majority_dataprovider=self.train_majority_dataprovider,
            train_minority_dataprovider=self.train_minority_dataprovider,
            valid_dataprovider=self.valid_dataprovider,
            valid_minority_dataprovider=self.valid_minority_dataprovider,
            attack=attack,
            num_epochs=self.num_epochs,
            optimizer=self.optimizer,
            results_dir=results_dir,
            scheduler=self.scheduler,
        )

        pass

    def experiment_one(self,results_dir):

        '''
        STEP 2: initialize the adversarial attack that will be used during adversarial training. 
        '''
        # results_dir = os.path.join(ROOT_DIR,'ExperimentResults/cifar10_advers_exp1')
        # attack hyperparameters:

        epsilon_attack = 4/255 # should probably be 4 (not 255).

        sys.stderr.write("creating attack.\n")
        attack = LInfProjectedGradientAttack(
            model=self.model,
            steps=1, alpha=0.01, epsilon=epsilon_attack, rand=True, targeted=False  # steps = 40 before, and alpha = 0.01
        )

        self.init_dataproviders(
            majority_batch_size=64,
            minority_batch_size=6,
            minority_percentage=1,
            which_data = 'cifar'
        )

        self.model.advers_train_and_evaluate(
            train_majority_dataprovider = self.train_majority_dataprovider,
            train_minority_dataprovider = self.train_minority_dataprovider,
            valid_dataprovider = self.valid_dataprovider,
            valid_minority_dataprovider = self.valid_minority_dataprovider,
            attack=attack,
            num_epochs=self.num_epochs,
            optimizer=self.optimizer,
            results_dir=results_dir,
            scheduler=self.scheduler,
        )

    def experiment_two(self,minority_percentage,results_dir):
        '''
        1. epsilon of attack is different from experiment one.
        2. minority percentage is different from experiment one.
        '''

        epsilon_attack = 4 # changed to 4; was 4/255. note cifar10 images have pixels between 0 and 255.
        sys.stderr.write("creating attack.\n")
        attack = LInfProjectedGradientAttack(
            model=self.model,
            steps=1, alpha=0.01, epsilon=epsilon_attack, rand=True, targeted=False  # steps = 40 before, and alpha = 0.01
        )
        self.init_dataproviders(
            majority_batch_size=64,
            minority_batch_size=6,
            minority_percentage=minority_percentage,
            which_data='cifar10'
        )
        self.model.advers_train_and_evaluate(
            train_majority_dataprovider = self.train_majority_dataprovider,
            train_minority_dataprovider = self.train_minority_dataprovider,
            valid_dataprovider = self.valid_dataprovider,
            valid_minority_dataprovider = self.valid_minority_dataprovider,
            attack=attack,
            num_epochs=self.num_epochs,
            optimizer=self.optimizer,
            results_dir=results_dir,
            scheduler=self.scheduler,
        )

if __name__ == '__main__':
    '''
    todo:
    1. training_acc target. (done)
    2. parsing arguments from command line. (done)
    3. test code on mnist. (still doing that)
    '''

    # e = AdversExperiment(which_model='mnist_simple',use_gpu=False)
    # e.num_epochs = 10
    # e.mnist_test_exp(
    #     results_dir=os.path.join(ROOT_DIR,'results/mnist_test_exp')
    # )

    e = AdversExperiment(which_model='densenet',use_gpu=True)
    e.experiment_two(
        minority_percentage=sys.argv[1],
        results_dir=os.path.join(ROOT_DIR,sys.argv[2])
    )

    '''
    STEP 3: specify what the minority class in the dataset is. minority class data is the data that we
    will be adversarially attacking.
    '''
    '''
    STEP 4: create a data provider for the training and validation set. the code below uses MNIST data.
    note: ImageDataIO is a convenience class for loading specific data into memory into the following format:
    x numpy array, shape (batch_size, num_channels, height, width), y numpy array, shape (batch_size,).
    '''

    '''
    STEP 5: call the "advers_train_and_evaluate" function that "model" has. this function takes a bunch of parameters.
    most arguments are familiar. the one that is different is "advs_images_file". this is the pickle file that adversarial
    images are saved to every epoch. recall how Antreas said it would be nice to see the progression of adversarial images
    created after each epoch of training our model. this is the file that saves those images.
    
    the contents of "advs_images_file" can be extrated using pickle. the content is a dictionary of the form:
    
    dict[epoch_num] = x,  where x is a numpy array of shape (num_channels, height, width) that corr. to an image.
    '''













