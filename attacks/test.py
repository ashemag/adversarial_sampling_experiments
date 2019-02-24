from adversarial_sampling_experiments.attacks.advers_attacks import *
from adversarial_sampling_experiments.attacks.data_augmenter import *
from adversarial_sampling_experiments.data_providers import *
from adversarial_sampling_experiments.models.simple_fnn import *
import torch.optim


def test_advers_attack_small():
    x, y = ImageDataGetter.mnist(filename=os.path.join(globals.ROOT_DIR, 'data/mnist-train.npz'))  # (1)
    data_iterator = DataProvider(x, y, batch_size=16, max_num_batches=1, make_one_hot=False, rng=None)  # (2)
    x, y = data_iterator.__next__()

    model = FeedForwardNetwork(img_shape=(1, 28, 28), num_classes=10)
    model.load_model(
        model_path=os.path.join(globals.ROOT_DIR, 'saved_models/simple_fnn/model_epoch_49')  # (3)
    )

    attack = LInfProjectedGradientAttack(
        model=model,
        steps=40,
        alpha=0.01,  # step size
        epsilon=0.3,
        rand=True,  # initialize at uniformly random feasible point
        targeted=False
    )

    x_adv = DataAugmenter.advers_attack(x, y, attack=attack)

    from adversarial_sampling_experiments.data_viewer import ImageDataViewer

    plot_dict = {}
    axs = plt.subplots(4, 4)
    axs = np.reshape(axs,(-1, ))
    labels = [y[i] for i in range(len(y))]
    for i in range(len(axs)):
        plot_dict[labels[i]] = {'ax': axs[i], 'img': x_adv[i]}

    ImageDataViewer.grid(plot_dict)

test_advers_attack_small()


def test_advers_train():
    model = FeedForwardNetwork(img_shape=(1, 28, 28), num_classes=10)

    x_aug, y = DataAugmenter.load_data(
        filename = os.path.join(globals.ROOT_DIR,'data/mnist_augmented/mnist-train-aug.npz')
    )

    # y has shape (batch_size,)

    aug_train_data = DataProvider(x_aug, y, batch_size=100, max_num_batches=1, make_one_hot=False, rng=None)

    model.train_full(
        train_data=aug_train_data,
        num_epochs=10,
        optimizer=optim.SGD(model.parameters(),lr=1e-1),
        train_file_path=os.path.join(globals.ROOT_DIR,'ExperimentResults/simple_fnn/train_results_aug.txt'),
        model_save_dir=os.path.join(globals.ROOT_DIR,'saved_models/simple_fnn_aug')
    )

def test_creating_advers_examples(targets=None,model=None,attack=None,save_path=None):
    if targets is None:
        x, y = ImageDataGetter.mnist(
            filename=os.path.join(globals.ROOT_DIR, 'data/mnist-train.npz')
        )
    else:
        y = targets

    if model is None:
        model = FeedForwardNetwork(img_shape=(1, 28, 28), num_classes=10)
        model.load_model(
            model_path=os.path.join(globals.ROOT_DIR,'saved_models/simple_fnn/model_epoch_49')
        )

    if attack is None:
        attack = LInfProjectedGradientAttack(
            model = model,
            steps = 10,
            alpha = 1,
            epsilon = 1,
            rand = True,
            targeted=False
        )

    x_adv = DataAugmenter.advers_attack(x,y,attack=attack)

    if save_path is not None:
        DataAugmenter.save_data(
            x_aug=x_adv, y=y,
            filename_npz = save_path # os.path.join(globals.ROOT_DIR, 'data/mnist_augmented/mnist-train-aug.npz')
        )

    return x_adv

def test_attack_advers_trained_model():
    '''
    what do adversarial images look like on network that has been trained to be robust against attacks?
    '''

    model = FeedForwardNetwork(img_shape=(1, 28, 28), num_classes=10)
    model.load_model(
        model_path=os.path.join(globals.ROOT_DIR, 'saved_models/simple_fnn_aug/model_epoch_49')
    )

    _, y = ImageDataGetter.mnist(
        filename=os.path.join(globals.ROOT_DIR, 'data/mnist-train.npz')
    )

    x_adv = test_creating_advers_examples(
        targets=y, # default: mnist targets
        model=model,
        attack=None, # use default attack
        save_path=None # doesn't save results
    )


if __name__ == '__main__':
    # test_creating_advers_examples()
    # test_advers_attack_small()

    from adversarial_sampling_experiments.globals import ROOT_DIR

    model = FeedForwardNetwork(img_shape=(1, 28, 28), num_classes=10)
    x, y = ImageDataGetter.mnist(filename=os.path.join(ROOT_DIR,'data/mnist-train.npz'))
    train_data_iterator = DataProvider(x,y,batch_size=100,max_num_batches=100,make_one_hot=False,rng=None)

    model.train_full(
        train_data=train_data_iterator,
        num_epochs=50,
        optimizer=optim.SGD(model.parameters(), lr=1e-1),
        train_file_path=os.path.join(globals.ROOT_DIR, 'ExperimentResults/simple_fnn/train_results.txt'),
        model_save_dir=os.path.join(globals.ROOT_DIR, 'saved_models/simple_fnn')
    )



