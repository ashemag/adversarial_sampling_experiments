from data_providers import *
from models.densenet import *
from globals import ROOT_DIR
import csv
from torchvision import transforms
import argparse
import torch.optim as optim
from models.base import *

BATCH_SIZE = 64
LEARNING_RATE = .1
WEIGHT_DECAY = 1e-4
MOMENTUM = .9

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def get_transform(set_name):
    if set_name == 'train':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


def get_args():
    parser = argparse.ArgumentParser(description='Minority class data experiment.')
    parser.add_argument('--label')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--target_percentage', type=float, default=-1)
    parser.add_argument('--full_flag', type=bool, default=False)  # full or reduced

    args = parser.parse_args()
    arg_str = [(str(key), str(value)) for (key, value) in vars(args).items()]
    print(arg_str)
    return args


def prepare_data(full_flag=False, minority_class=3, minority_percentage=0.01):
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
    valid_data = torch.utils.data.DataLoader(valid_set, batch_size=64, shuffle=True, num_workers=4)

    # LOAD TEST DATA
    test_set = CIFAR10(root='data', transform=get_transform('valid'), download=True, set_name='test',
                       percentages_list=percentages)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=4)

    # REDUCE MINORITY CLASS IN TRAINING
    minority_class_idx = -1
    if not full_flag:
        percentages[minority_class] = minority_percentage
        minority_class_idx = minority_class

    train_set = CIFAR10(root='data', transform=get_transform('train'), download=True, set_name='train',
                        percentages_list=percentages)
    train_data = MinorityDataLoader(train_set, batch_size=64, shuffle=True, num_workers=4,
                                    minority_class_idx=minority_class_idx)

    return train_data, valid_data, test_data


def get_label_mapping():
    d = unpickle('data/cifar-10-batches-py/batches.meta')
    labels = d[b'label_names']
    return {value.decode('ascii'): index for index, value in enumerate(labels)}


def prepare_output_file(filename, output=None, clean_flag=False):
    file_exists = os.path.isfile(filename)
    if clean_flag:
        if file_exists:
            os.remove(filename)
    else:
        if output is None:
            raise ValueError("Please specify output to write to output file.")
        with open(filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(output.keys()))
            if not file_exists:
                writer.writeheader()
            print("Writing to file {0}".format(filename))
            print(output)
            writer.writerow(output)


if __name__ == "__main__":
    args = get_args()
    if args.full_flag:
        model_title = args.label + '_full_' + str(args.seed)
    else:
        model_title = args.label + '_' + str(args.target_percentage) + '%_' + str(args.seed)
    target_percentage = args.target_percentage / 100
    print("Running {0}".format(model_title))

    # SET RANDOMNESS
    torch.manual_seed(seed=args.seed)
    rng = np.random.RandomState(args.seed)
    label_mapping = get_label_mapping()

    # TRUE WHEN STARTING COMPLETELY NEW EXPERIMENT
    train_data, valid_data, test_data = prepare_data(full_flag=args.full_flag,
                                                     minority_class=label_mapping[args.label],
                                                     minority_percentage=target_percentage)

    #OUTPUT
    results_dir = os.path.join(ROOT_DIR, 'results/{}').format(model_title)

    # EXPERIMENT
    model = DenseNet121()
    model = model.to(model.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                                momentum=MOMENTUM,
                                nesterov=True,
                                weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0.0001)


    bpm_overall, bpm_minority = model.train_evaluate(
        train_set=train_data,
        valid_full=valid_data,
        test_full=test_data,
        num_epochs=args.num_epochs,
        optimizer=optimizer,
        results_dir=results_dir,
        scheduler=scheduler,
        minority_class=label_mapping[args.label], #  type int
    )

    bpm_overall['model_title'] = model_title
    bpm_minority['model_title'] = model_title

    output_dir_overall = os.path.join(ROOT_DIR, 'data/minority_class_experiments_bpm_overall.csv')
    output_dir_minority = os.path.join(ROOT_DIR, 'data/minority_class_experiments_bpm_minority.csv')

    prepare_output_file(output=bpm_overall, filename=output_dir_overall)
    prepare_output_file(output=bpm_minority, filename=output_dir_minority)
