from models.densenet import DenseNet121
import numpy as np
from data_subsetter import DataSubsetter
from data_io import ImageDataIO
import torch
from torch import optim
import os
from globals import ROOT_DIR
import sys
from attacks.advers_attacks import LInfProjectedGradientAttack
from data_providers import DataProvider
from models.base import Logger

logger = Logger(stream = sys.stderr,disable= False)

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

class TrainSamplerSimple():
    def __init__(self,train_data,minority_batch_size,majority_batch_size,labels_majority,labels_minority,
                 minority_reduction_factor):
        self.minority_batch_size = minority_batch_size
        self.majority_batch_size = majority_batch_size
        self.labels_majority = labels_majority
        self.labels_minority = labels_minority
        self.minority_reduction_factor = minority_reduction_factor
        self.mino_sampler, self.maj_sampler = self.init_samplers(train_data)


    def init_samplers(self,train_data):
        x, y = train_data
        x_maj, y_maj = DataSubsetter.condition_on_label(x, y, labels=self.labels_majority, shuffle=True, rng=None)
        x_mino, y_mino = DataSubsetter.condition_on_label(x, y, labels=self.labels_minority, shuffle=True, rng=None)
        size_full_minority = len(x_mino)
        size_minority = int(len(x_mino) * self.minority_reduction_factor)
        x_mino = x_mino[:size_minority]
        y_mino = y_mino[:size_minority]

        '''
        def __init__(self, inputs, targets, batch_size, max_num_batches=-1,
                             shuffle_order=True, rng=None, make_one_hot=True, with_replacement=False):
        '''

        mino_sampler = DataProvider(
            inputs=x_mino,
            targets=y_mino,
            batch_size=self.minority_batch_size,
            shuffle_order=True,
            rng=None,
            make_one_hot=False,
            with_replacement=True
        )

        maj_sampler = DataProvider(
            inputs=x_maj,
            targets=y_maj,
            batch_size=self.majority_batch_size,
            shuffle_order=True,
            rng=None,
            make_one_hot=False,
            with_replacement=False
        )

        logger.print("minority reduced size: {}. minority size: {}. majority size: {}"
              .format(size_minority, size_full_minority, len(x_maj)))
        logger.print("percentage: {}".format(self.minority_reduction_factor))

        return mino_sampler, maj_sampler

    def __iter__(self):  # implement iterator interface.
        return self

    def __next__(self):
        try:
            x_maj_batch, y_maj_batch = next(self.maj_sampler)
            x_mino_batch, y_mino_batch = next(self.mino_sampler)
            batch = (x_maj_batch,y_maj_batch,x_mino_batch,y_mino_batch)
            return batch
        except:
            raise StopIteration()

class TrainSampler():
    def __init__(self,train_data, minority_mean_batch_size, majority_mean_batch_size, labels_minority,
                 labels_majority, minority_reduction_factor=1.):

        self.minority_mean_batch_size = minority_mean_batch_size
        self.majority_mean_batch_size = majority_mean_batch_size
        self.total_batch_size = int(minority_mean_batch_size + majority_mean_batch_size)
        print("total batch size: ",self.total_batch_size)

        self.minority_reduction_factor = minority_reduction_factor
        self.labels_minority = labels_minority
        self.labels_majority = labels_majority
        self.mino_sampler, self.maj_sampler = self.init_samplers(train_data)

        frac = self.majority_mean_batch_size / self.total_batch_size
        self.batch_sizes = np.random.binomial(self.total_batch_size, p=frac, size=(2000,))
        self.majority_batch_size_idx = 0

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
        majority_batch_size = self.batch_sizes[self.majority_batch_size_idx]
        self.majority_batch_size_idx += 1

        minority_batch_size = self.total_batch_size - majority_batch_size

        if self.maj_sampler.has_next():
            x_maj_batch, y_maj_batch = \
                self.maj_sampler.sample(batch_size=majority_batch_size)
            x_mino_batch, y_mino_batch = \
                self.mino_sampler.sample(batch_size=minority_batch_size)
            batch = (x_maj_batch, y_maj_batch, x_mino_batch, y_mino_batch)
            return batch
        else:
            raise StopIteration()

class TestSampler(object):
    def __init__(self,data,labels_minority):
        self.x, self.y = data
        self.x_mino, self.y_mino = DataSubsetter.condition_on_label(self.x, self.y, labels=labels_minority, shuffle=True, rng=None)
        self.has_next = True

    def __iter__(self):
        return self

    def __next__(self): # data is given in one go.
        if self.has_next:
            self.has_next = False
            return self.x,self.y,self.x_mino,self.y_mino
        else:
            self.has_next = True # reset.
            raise StopIteration()

class TestSamplerSimple(object):
    def __init__(self,data,labels_minority):
        self.x, self.y = data
        self.x_mino, self.y_mino = DataSubsetter.condition_on_label(self.x, self.y, labels=labels_minority, shuffle=True, rng=None)

        self.full_sampler = DataProvider(
            inputs=self.x,
            targets=self.y,
            batch_size=100,
            shuffle_order=True,
            rng=None,
            make_one_hot=False,
            with_replacement=False
        )

        self.mino_sampler = DataProvider(
            inputs=self.x_mino,
            targets=self.y_mino,
            batch_size=100,
            shuffle_order=True,
            rng=None,
            make_one_hot=False,
            with_replacement=False
        )

    # def __iter__(self):
    #     return self
    #
    # def __next__(self): # data is given in one go.
    #     try:
    #         x, y = next(self.full_sampler)
    #         x_mino, y_mino = next(self.mino_sampler)
    #         batch = (x,y,x_mino,y_mino)
    #         return batch
    #     except:
    #        raise StopIteration()

def mnist_experiment():
    minority_class = 3
    x_train, y_train = ImageDataIO.mnist('train')
    x_valid, y_valid = ImageDataIO.mnist('valid')
    x_test, y_test = ImageDataIO.mnist('test')

    x_train = x_train[:4000]
    y_train = y_train[:4000]

    train_sampler = TrainSampler(
        train_data=(x_train, y_train),
        minority_mean_batch_size=64 * 0.1,
        majority_mean_batch_size=64 * 0.9,
        labels_minority=[minority_class],  # cat
        labels_majority=[i for i in range(10) if i != minority_class],
        minority_reduction_factor=1,  # (minority percentage)
    )
    valid_sampler = TestSampler(
        data=(x_valid, y_valid),
        labels_minority=[minority_class]
    )
    test_sampler = TestSampler(
        data=(x_test, y_test),
        labels_minority=[minority_class]
    )

    from models.simple_fnn import FeedForwardNetwork
    config_list = [
        {'type': 'fc', 'out_features': 50, 'bias': False, 'nl': 'relu', 'dropout': None},
        {'type': 'fc', 'out_features': 20, 'bias': False, 'nl': 'relu', 'dropout': None}
    ]
    model = FeedForwardNetwork(img_shape=(1, 28, 28), num_classes=10, config_list=config_list)

    attack = LInfProjectedGradientAttack(
        model=model,
        steps=1,
        alpha=1,
        epsilon=0.3,
        rand=True,
        targeted=False
    )

    LEARNING_RATE = .1
    WEIGHT_DECAY = 1e-4
    MOMENTUM = .9
    num_epochs = 20

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True,
                                weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0001)

    model.advers_train_and_evaluate_uniform(
        train_sampler=train_sampler,
        valid_sampler=valid_sampler,
        test_sampler=test_sampler,
        attack=attack,
        num_epochs=num_epochs,
        optimizer=optimizer,
        results_dir=os.path.join(ROOT_DIR,'results/final_mnist_test'),
        scheduler=scheduler
    )

def cifar_experiment_rotated_attack():
    from attacks.advers_attacks import RotateAttack

    # TODO: Make sure you re-download the data!

    minority_class = 3
    x_train, y_train = ImageDataIO.cifar10('train', normalize=True)
    x_valid, y_valid = ImageDataIO.cifar10('valid', normalize=True)
    x_test, y_test = ImageDataIO.cifar10('test', normalize=True)

    num_obs = 2000
    x_train = x_train[:num_obs]
    y_train = y_train[:num_obs]

    train_sampler = TrainSamplerSimple(
        train_data=(x_train, y_train),
        minority_batch_size=6,
        majority_batch_size=64,
        labels_minority=[minority_class],  # cat
        labels_majority=[i for i in range(10) if i != minority_class],
        minority_reduction_factor=1,  # (minority percentage)
    )

    valid_sampler = TestSamplerSimple(
        data=(x_valid, y_valid),
        labels_minority=[minority_class]
    )

    test_sampler = TestSamplerSimple(
        data=(x_test, y_test),
        labels_minority=[minority_class]
    )

    model = DenseNet121()
    model.use_gpu(gpu_ids='0')

    attack = RotateAttack() # no arg means uses default rotations.

    # Note: Slight problem in code: code assumes attack.model exists!

    LEARNING_RATE = .1
    WEIGHT_DECAY = 1e-4
    MOMENTUM = .9
    num_epochs = 120

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True,
                                weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0001)

    model.advers_train_and_evaluate_uniform(
        train_sampler=train_sampler,
        valid_sampler=valid_sampler,
        test_sampler=test_sampler,
        attack=attack,
        num_epochs=num_epochs,
        optimizer=optimizer,
        results_dir=os.path.join('results/final_cifar10_test'),
        scheduler=scheduler
    )

    pass

def cifar_experiment(minority_percentage,results_dir, advers=False, rotated_attack=False, epsilon=40 / 255):
    from attacks.advers_attacks import RotateAttack

    minority_class = 3
    x_train, y_train = ImageDataIO.cifar10('train',normalize=True)
    x_valid, y_valid = ImageDataIO.cifar10('valid',normalize=True)
    x_test, y_test = ImageDataIO.cifar10('test',normalize=True)

    num_obs = len(x_train)
    x_train = x_train[:num_obs]
    y_train = y_train[:num_obs]

    # NOTE: Why did I make these changes of the validation set? Because of memory issues - I forward propagate with
    # batches now instead and it works!

    train_sampler = TrainSamplerSimple(
        train_data=(x_train,y_train),
        minority_batch_size=6,
        majority_batch_size=64,
        labels_minority=[minority_class],  # cat
        labels_majority=[i for i in range(10) if i != minority_class],
        minority_reduction_factor=minority_percentage,  # (minority percentage)
    )

    train_sampler = TrainSampler(
        train_data=(x_train, y_train),
        minority_mean_batch_size=64 * 0.1,
        majority_mean_batch_size=64 * 0.9,
        labels_minority=[minority_class],  # cat
        labels_majority=[i for i in range(10) if i != minority_class],
        minority_reduction_factor=minority_percentage,  # (minority percentage)
    )


    valid_sampler = TestSamplerSimple(
        data=(x_valid,y_valid),
        labels_minority=[minority_class]
    )

    test_sampler = TestSamplerSimple(
        data=(x_test,y_test),
        labels_minority=[minority_class]
    )

    model = DenseNet121()
    model.use_gpu(gpu_ids='0')

    if advers:
        attack = LInfProjectedGradientAttack(
            model=model,
            steps=1,
            alpha=1,
            epsilon=epsilon,
            rand=True,
            targeted=False
        )
    elif rotated_attack:
        attack = RotateAttack(
            model=model # needs model to put on correct device.
        )
    else:
        raise Exception('you did not specify an attack.')


    LEARNING_RATE = .1
    WEIGHT_DECAY = 1e-4
    MOMENTUM = .9
    num_epochs = 120

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True,weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0001)

    model.advers_train_and_evaluate_uniform_tens(
        train_sampler = train_sampler,
        valid_sampler = valid_sampler,
        test_sampler = test_sampler,
        attack = attack,
        num_epochs = num_epochs,
        optimizer = optimizer,
        results_dir=results_dir,
        scheduler=scheduler
    )

if __name__ == '__main__':
    # mnist_experiment()
    minority_percentage = 0.01
    results_dir = os.path.join('final_results/rotated_attack_{}'.format(minority_percentage))
    cifar_experiment(
        results_dir=results_dir,
        minority_percentage=minority_percentage,
        rotated_attack=True
    )
