from comet_ml import Experiment

import time
from data_providers import *
from models.densenet import *
from globals import ROOT_DIR
import torch.optim as optim
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from experiment_builder import ExperimentBuilder
from experiment_utils import (set_device,
                              get_cifar_labels_to_ints,
                              get_cifar_ints_to_labels,
                              get_args,
                              print_duration,
                              get_transform)
BATCH_SIZE = 2048
LEARNING_RATE = .1
WEIGHT_DECAY = 1e-4
MOMENTUM = .9


def prepare_data(full_flag, batch_size, minority_class=3, minority_percentage=0.01):
    """

    :param full_flag: if we are reducing class or not
    :param minority_class: minority class idx (int)
    :param minority_percentage: percentage (float)
    :return:
    """
    percentages = [1. for i in range(10)]

    # LOAD VALID DATA
    valid_set = CIFAR10(root='data', transform=get_transform('valid'), download=True, set_name='val',
                        percentages_list=percentages)
    valid_data = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=4)

    # LOAD TEST DATA
    test_set = CIFAR10(root='data', transform=get_transform('valid'), download=True, set_name='test',
                       percentages_list=percentages)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)

    # REDUCE MINORITY CLASS IN TRAINING
    minority_class_idx = -1
    if not full_flag:
        percentages[minority_class] = minority_percentage
        minority_class_idx = minority_class

    train_set = CIFAR10(root='data', transform=get_transform('train'), download=True, set_name='train',
                        percentages_list=percentages)
    train_data = MinorityDataLoader(train_set,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=4,
                                    minority_class_idx=minority_class_idx)
    print("Train: {}, Valid: {}, Test: {}".format(len(train_set), len(valid_set), len(test_set)))
    return train_data, valid_data, test_data


if __name__ == "__main__":
    start_time = time.time()
    args = get_args()
    target_percentage = args.target_percentage / 100

    # Create experiment name and experiment folder
    experiment_name = '_'.join([args.label, str(args.target_percentage), str(args.seed)])
    experiment_folder = os.path.join(ROOT_DIR, 'results/{}').format(experiment_name)
    print("=== Experiment ===\n{}".format(experiment_name))

    # Fetch data components
    labels_to_ints = get_cifar_labels_to_ints()
    ints_to_labels = get_cifar_ints_to_labels()
    train_data, valid_data, test_data = prepare_data(full_flag=args.full_flag,
                                                     batch_size=args.batch_size,
                                                     minority_class=labels_to_ints[args.label],
                                                     minority_percentage=target_percentage)

    # Fetch model components
    model = DenseNet121()
    device = set_device(args.seed)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=LEARNING_RATE,
                                momentum=MOMENTUM,
                                nesterov=True,
                                weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0.0001)

    # comet experiment
    comet_experiment = Experiment(project_name="minority-experiments")
    comet_experiment.set_name(experiment_name)

    # Run experiment
    experiment = ExperimentBuilder(
        model=model,
        device=device,
        label_mapping=ints_to_labels,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        optimizer=optimizer,
        scheduler=scheduler,
        experiment_folder=experiment_folder,
        comet_experiment=comet_experiment
    )

    experiment.run_experiment(num_epochs=args.num_epochs, seed=args.seed, experiment_name=experiment_name)
    print("=== Total experiment runtime ===")
    print_duration(time.time() - start_time)
