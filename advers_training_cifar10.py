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
from models.simple_fnn import FeedForwardNetwork
import numpy as np
logger = Logger(stream=sys.stderr, disable=False)

class Sampler(object):
    DEFAULT_SEED = 20112018

    def __init__(self,x,y,batch_size=None,rng=None,shuffle_first=True):
        self.x = x
        self.y = y
        self.x_copy = x
        self.y_copy = y
        self.shuffle_first = shuffle_first
        if rng is None: self.rng = np.random.RandomState(Sampler.DEFAULT_SEED)
        if batch_size is None: self.batch_size = len(x)

    def __iter__(self):  # implement iterator interface.
        return self

    def __next__(self):
        return self.sample()

    def sample(self,batch_size=None): # this can be overriden to change sampling behavior.
        if batch_size is not None: self.batch_size = batch_size
        if self.shuffle_first: self.shuffle()
        if self.has_next():
            xx = self.x[:self.batch_size]  # return top.
            yy = self.y[:self.batch_size]
            self.x = self.x[self.batch_size:]  # remove top
            self.y = self.y[
                     self.batch_size:]  # note: if 0 < len(x) < batch_size then x[batch_size:] returns [] (is len 0)
            batch = (xx,yy)
            return batch
        else:
            raise StopIteration()

    def has_next(self):
        if len(self.x)>0:
            return True
        self.reset()  # fill up again for next time you want to iterate over it.
        return False

    def reset(self):
        self.x = self.x_copy
        self.y = self.y_copy

    def shuffle(self):
        perm = self.rng.permutation(len(self.x))
        return self.x[perm], self.y[perm]

class SampleWithReplacement(Sampler):
    def __init__(self,x,y,batch_size=None,rng=None,shuffle_first=True):
        super(SampleWithReplacement, self).__init__(x,y,batch_size,rng,shuffle_first)

    def sample(self,batch_size=None):
        if batch_size is not None: self.batch_size = batch_size
        if self.shuffle_first: self.shuffle()
        if self.has_next():
            xx = self.x[:self.batch_size]  # return top.
            yy = self.y[:self.batch_size]
            batch = (xx,yy)
            return batch
        else:
            raise StopIteration()

class AdversTrainSampler():
    def __init__(self,train_data, minority_mean_batch_size, majority_mean_batch_size, labels_minority,
                 labels_majority, minority_reduction_factor=1):

        self.minority_mean_batch_size = minority_mean_batch_size
        self.majority_mean_batch_size = majority_mean_batch_size
        self.minority_reduction_factor = minority_reduction_factor
        self.labels_minority = labels_minority
        self.labels_majority = labels_majority
        self.mino_sampler, self.maj_sampler = self.init_samplers(train_data)

    def init_samplers(self, train_data):
        x, y = train_data
        x_maj, y_maj = DataSubsetter.condition_on_label(x, y, labels=self.labels_majority, shuffle=True, rng=None)
        x_mino, y_mino = DataSubsetter.condition_on_label(x, y, labels=self.labels_minority, shuffle=True, rng=None)
        size_full_minority = len(x_mino)
        size_minority = int(len(x_mino) * self.minority_reduction_factor)
        x_mino = x_mino[:size_minority]
        y_mino = y_mino[:size_minority]

        mino_sampler = SampleWithReplacement(x_mino,y_mino,batch_size=None,rng=None,shuffle_first=True)
        maj_sampler = Sampler(x_maj, y_maj,batch_size=None,rng=None,shuffle_first=True)
        print("minority reduced size: {}. minority size: {}. majority size: {}"
                     .format(size_minority, size_full_minority, len(x_maj)))
        print("percentage: {}".format(self.minority_reduction_factor))

        return mino_sampler, maj_sampler

    def __iter__(self):  # implement iterator interface.
        return self

    def __next__(self):
        total_batch_size = self.minority_mean_batch_size + self.majority_mean_batch_size
        frac = self.majority_mean_batch_size / total_batch_size
        majority_batch_size = np.random.binomial(total_batch_size, p=frac)
        minority_batch_size = total_batch_size - majority_batch_size
        if self.maj_sampler.has_next():
            x_maj_batch, y_maj_batch = \
                self.maj_sampler.sample(batch_size=majority_batch_size)
            x_mino_batch, y_mino_batch = \
                self.mino_sampler.sample(batch_size=minority_batch_size)
            batch = (x_maj_batch, y_maj_batch, x_mino_batch, y_mino_batch)
            return batch
        else:
            raise StopIteration()


class AdversExperiment():
    def __init__(self,which_model=None,use_gpu=True,model=None):
        self.labels_minority = [8]  # must be a list.
        self.labels_majority = [i for i in range(10) if i not in [8]]
        self.num_epochs = 120
        BATCH_SIZE = 64
        LEARNING_RATE = .1
        WEIGHT_DECAY = 1e-4
        MOMENTUM = .9
        if model is None:
            self.model = AdversExperiment.get_model(which_model=which_model) # densenet for cifar10
        else:
            self.model = model
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
            config_list = [
                {'type': 'fc', 'out_features': 50, 'bias': False, 'nl': 'relu', 'dropout': None},
                {'type': 'fc', 'out_features': 20, 'bias': False, 'nl': 'relu', 'dropout': None}
            ]

            model = FeedForwardNetwork(img_shape=(1, 28, 28), num_classes=10, config_list=config_list)
        return model

    def create_providers_normal(self,batch_size,which_data):
        if which_data=='cifar':
            x_train, y_train = ImageDataIO.cifar10('train')
            x_test, y_test = ImageDataIO.cifar10('test')
        elif which_data == 'mnist':
            x_train, y_train = ImageDataIO.mnist('train')
            x_test, y_test = ImageDataIO.mnist('test')
        else:
            return

        train_dataprovider = DataProvider(
            inputs=x_train,
            targets=y_train,
            batch_size=batch_size,
            max_num_batches=-1,
            make_one_hot=False,
            rng=None,
            with_replacement=False
        )

        valid_dataprovider = DataProvider(
            inputs=x_test,
            targets=y_test,
            batch_size=100, # doesn't matter.
            max_num_batches=-1,
            make_one_hot=False,
            rng=None,
            with_replacement=False
        )

        logger.print('################ data information ################')
        logger.print('train size: {}'.format(len(x_train)))
        logger.print('test size: {}'.format(len(x_test)))
        logger.print('##################################################')

        return train_dataprovider, valid_dataprovider

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



    def advers_train_normal_mnist(self,results_dir):
        epsilon_attack = 0.3 # 4 / 255  # pixels in mnist data are between 0 and 1 i think (verify)

        logger.print('creating attack.')
        attack = LInfProjectedGradientAttack(
            model=self.model,
            steps=1, alpha=0.01, epsilon=epsilon_attack, rand=True, targeted=False
        )

        train_dataprovider, valid_dataprovider = self.create_providers_normal(batch_size=100,which_data='mnist')

        self.num_epochs = 20
        self.model.advers_train_normal(
            train_dataprovider=train_dataprovider,
            valid_dataprovider=valid_dataprovider,
            attack=attack,
            num_epochs=self.num_epochs,
            optimizer=self.optimizer,
            results_dir=results_dir,
            scheduler=self.scheduler
        )

    def normal_train_mnist(self,results_dir):
        train_dataprovider, valid_dataprovider = self.create_providers_normal(batch_size=100, which_data='mnist')

        self.num_epochs = 20
        self.model.normal_train(
            train_dataprovider=train_dataprovider,
            valid_dataprovider=valid_dataprovider,
            num_epochs=self.num_epochs,
            optimizer=self.optimizer,
            results_dir=results_dir,
            scheduler=self.scheduler
        )

from data_viewer import ImageDataViewer
import matplotlib.pyplot as plt

class SanityChecks():

    @staticmethod
    def mnist_view_data():
        x, y = ImageDataIO.mnist('train')
        num_images = 10
        labels = [i for i in range(num_images)]
        x = x[:num_images]
        cmap = plt.cm.get_cmap('Greens') # grey_r
        ImageDataViewer.batch_view(x,nrows=2,ncols=5,labels=labels,cmap=cmap)

    @staticmethod
    def attack_with_empty_model():
        model = AdversExperiment.get_model('mnist_simple')
        epsilon_attack = 0.3 # 4/255 # 4 / 255  # pixels in mnist data are between 0 and 1 i think (verify)
        logger.print('creating attack.')
        alpha = 0.01

        attack = LInfProjectedGradientAttack(
            model=model,
            steps=1, alpha=alpha, epsilon=epsilon_attack, rand=True, targeted=False
        )
        x, y = ImageDataIO.mnist('train')
        num_images = 10
        x = x[:num_images]
        y = y[:num_images]
        labels = [i for i in range(num_images)]
        x_adv = attack(x,y)
        cmap = plt.cm.get_cmap('Greens')  # grey_r
        ImageDataViewer.batch_view(x_adv, nrows=2, ncols=5, labels=labels, cmap=cmap)


    @staticmethod
    def attack_test_performance():
        from models.simple_fnn import FeedForwardNetwork
        config_list = [
            {'type': 'fc', 'out_features': 50, 'bias': False, 'nl': 'relu', 'dropout': None},
            {'type': 'fc', 'out_features': 20, 'bias': False, 'nl': 'relu', 'dropout': None}
        ]

        ''' WHAT TYPE OF MODEL?
        1. Empty model.
        2. Normally trained model.
        3. Adversarially trained starting with empty.
        4. Adversarially trained starting from trained model. 
        '''

        empty_model = FeedForwardNetwork(img_shape=(1, 28, 28), num_classes=10, config_list=config_list)
        # normal_trained_model = FeedForwardNetwork(img_shape=(1, 28, 28), num_classes=10, config_list=config_list)
        # advers_trained_model_from_empty = FeedForwardNetwork(img_shape=(1, 28, 28), num_classes=10, config_list=config_list)
        # advers_trained_model_from_trained = FeedForwardNetwork(img_shape=(1, 28, 28), num_classes=10, config_list=config_list)
        # for some reason when creating multiple FeedForwardNetworks it does something weird!
        # maybe static method is the problem?

        normal_trained_model = empty_model
        normal_trained_model.load_model(
            model_path= os.path.join(ROOT_DIR,'results/normal_model_mnist/model/model_epoch_19')
        )

        advers_trained_model_from_trained = empty_model
        advers_trained_model_from_trained.load_model(
            model_path=os.path.join(ROOT_DIR, 'results/advers_model_from_trained_mnist/model/model_epoch_19')
        )

        advers_trained_model_from_empty = empty_model
        advers_trained_model_from_empty.load_model(
            model_path= os.path.join(ROOT_DIR,'results/advers_model_from_empty_mnist/model/model_epoch_19')
        )

        def test_model_performance(x,y,model):
            acc  = model.get_acc_batch(x, y, y_batch_pred=None, integer_encoded=True)
            return acc

        x, y = ImageDataIO.mnist('test')
        result_line = ''
        model_names = ['empty_model','normal_trained_model','advers_trained_model_from_empty','advers_trained_model_from_trained']
        models = [empty_model,normal_trained_model,advers_trained_model_from_empty,advers_trained_model_from_trained]
        for model_name,model in zip(model_names,models):
            print(model_name, model)
            acc_of_model = test_model_performance(x,y,model)
            result_line = result_line + '{}: {}\n'.format(model_name,acc_of_model)

        print(result_line)

        # epsilon_attack = 0.3  # 4/255 # 4 / 255  # pixels in mnist data are between 0 and 1 i think (verify)
        # logger.print('creating attack.')
        # alpha = 0.01
        #
        # attack = LInfProjectedGradientAttack(
        #     model=model,
        #     steps=1, alpha=alpha, epsilon=epsilon_attack, rand=True, targeted=False
        # )
        # x, y = ImageDataIO.mnist('train')
        # num_images = 8
        # x = x[:num_images]
        # y = y[:num_images]
        #
        # def test(x):
        #     x = torch.Tensor(x).float().to(device='cpu')
        #     y_pred = model.forward(x)
        #     print("y pred data: ", y_pred.data.size())
        #
        #     _, y_pred_int = torch.max(y_pred.data, 1)
        #     y_pred_int = list(y_pred_int.numpy())
        #     return y_pred_int
        #
        # cmap = plt.cm.get_cmap('Greens')  # grey_r
        # y_pred_int = test(x)
        # labels = ['true: {}\n pred: {}'.format(y[i],y_pred_int[i]) for i in range(len(x))]
        #
        # ImageDataViewer.batch_view(x, nrows=4, ncols=2, labels=labels, cmap=cmap,hspace=0.1)
        #
        # x_adv = attack(x, y)
        # y_pred_int_adv = test(x_adv)
        # print(y_pred_int_adv)
        # labels = ['true: {}\n pred: {}'.format(y[i], y_pred_int_adv[i]) for i in range(len(x))]
        #
        # ImageDataViewer.batch_view(x_adv, nrows=4, ncols=2, labels=labels, cmap=cmap,hspace=0.1)


    @staticmethod
    def mnist_advers_attack():
        config_list = [
            {'type': 'fc', 'out_features': 50, 'bias': False, 'nl': 'relu', 'dropout': None},
            {'type': 'fc', 'out_features': 20, 'bias': False, 'nl': 'relu', 'dropout': None}
        ]
        model = FeedForwardNetwork(img_shape=(1, 28, 28), num_classes=10, config_list=config_list)

        # the model has to be trained - possible problem that is happening is that

        epsilon_attack = 4 / 255  # pixels in mnist data are between 0 and 1 i think (verify)

        logger.print('creating attack.')
        attack = LInfProjectedGradientAttack(
            model=model,
            steps=1, alpha=0.01, epsilon=epsilon_attack, rand=True, targeted=False
        )

class Visuals(object):

    @staticmethod
    def visualize_max_norm_attack():
        # normally trained model.

        target_class = [3] # cat.

        x, y = ImageDataIO.cifar10('train')


        x_train_majority, y_train_majority = \
            DataSubsetter.condition_on_label(x_train, y_train, labels=self.labels_majority, shuffle=True, rng=None)


        pass


def test_performance_models():
    config_list = [
        {'type': 'fc', 'out_features': 50, 'bias': False, 'nl': 'relu', 'dropout': None},
        {'type': 'fc', 'out_features': 20, 'bias': False, 'nl': 'relu', 'dropout': None}
    ]
    empty_model = FeedForwardNetwork(img_shape=(1, 28, 28), num_classes=10, config_list=config_list)
    normal_trained_model = FeedForwardNetwork(img_shape=(1, 28, 28), num_classes=10, config_list=config_list)
    advers_trained_model_from_empty = FeedForwardNetwork(img_shape=(1, 28, 28), num_classes=10, config_list=config_list)
    advers_trained_model_from_trained = FeedForwardNetwork(img_shape=(1, 28, 28), num_classes=10, config_list=config_list)

    normal_trained_model.load_model(
        model_path=os.path.join(ROOT_DIR, 'results/normal_model_mnist/model/model_epoch_19')
    )
    advers_trained_model_from_trained.load_model(
        model_path=os.path.join(ROOT_DIR,'results/advers_model_from_trained_mnist/model/model_epoch_19')
    )
    advers_trained_model_from_empty.load_model(
        model_path=os.path.join(ROOT_DIR, 'results/advers_model_from_empty_mnist/model/model_epoch_19')
    )

    def test_model_performance(x, y, model):
        acc = model.get_acc_batch(x, y, y_batch_pred=None, integer_encoded=True)
        return acc

    epsilon_attack = 0.3 # 3*0.3  # 4 / 255  # pixels in mnist data are between 0 and 1 i think (verify)

    logger.print('creating attack.')
    attack_normal = LInfProjectedGradientAttack(
        model=normal_trained_model,
        steps=1, alpha=0.1, epsilon=epsilon_attack, rand=True, targeted=False
    )

    attack_from_empty = LInfProjectedGradientAttack(
        model=advers_trained_model_from_empty,
        steps=1, alpha=0.1, epsilon=epsilon_attack, rand=True, targeted=False
    )

    x, y = ImageDataIO.mnist('test')
    x_adv = attack_normal(x, y)

    acc_of_model = test_model_performance(x_adv, y, normal_trained_model)
    print("adv acc normal: ", acc_of_model)

    acc_of_model = test_model_performance(x_adv, y, advers_trained_model_from_empty)
    print("adv acc from empty: ", acc_of_model)

    acc_of_model = test_model_performance(x_adv, y, advers_trained_model_from_trained)
    print("adv acc from trained: ", acc_of_model)

    # attack the advers trained models:

    x_adv_normal = attack_normal(x,y)
    x_adv_empty = attack_from_empty(x,y)

    num_images = 10
    x_adv_normal = x_adv_normal[:num_images]
    x_adv_empty = x_adv_empty[:num_images]

    labels = [i for i in range(num_images)]
    cmap = plt.cm.get_cmap('Greens')  # grey_r
    ImageDataViewer.batch_view(x_adv_normal, nrows=5, ncols=2, labels=labels, cmap=cmap,hspace=0.1)
    ImageDataViewer.batch_view(x_adv_empty, nrows=5, ncols=2, labels=labels, cmap=cmap, hspace=0.1)

if __name__ == '__main__':
    # test_performance_models()
    # 1. advers train from empty model. (done)
    # e = AdversExperiment(which_model='mnist_simple',use_gpu=False)
    # e.advers_train_normal_mnist(
    #     results_dir=os.path.join(ROOT_DIR,'results/advers_model_from_empty_mnist')
    # )

    # 2. train normal model.
    # e = AdversExperiment(which_model='mnist_simple', use_gpu=False)
    # e.normal_train_mnist(
    #     results_dir=os.path.join(ROOT_DIR, 'results/normal_model_mnist')
    # )

    # 3. train advers model from trained model. (done)
    # config_list = [
    #     {'type': 'fc', 'out_features': 50, 'bias': False, 'nl': 'relu', 'dropout': None},
    #     {'type': 'fc', 'out_features': 20, 'bias': False, 'nl': 'relu', 'dropout': None}
    # ]
    # trained_model = FeedForwardNetwork(img_shape=(1, 28, 28), num_classes=10, config_list=config_list)
    # trained_model.load_model(
    #     model_path= os.path.join(ROOT_DIR,'results/normal_model_mnist/model/model_epoch_19')
    # )
    #
    # e = AdversExperiment(model=trained_model,use_gpu=False)
    # e.advers_train_normal_mnist(
    #     results_dir=os.path.join(ROOT_DIR,'results/advers_model_from_trained_mnist')
    # )

    e = AdversExperiment(which_model='densenet',use_gpu=True)
    e.experiment_two(
        minority_percentage=float(sys.argv[1]),
        results_dir='ExperimentResults/cifar10_test.txt')


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













