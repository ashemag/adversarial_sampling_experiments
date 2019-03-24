from copy import copy

from data_providers import *
from models.densenet import *
import globals
import csv
from torchvision import transforms
import argparse
from sklearn.model_selection import train_test_split
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
    parser.add_argument('--target_percentage', type=int)
    parser.add_argument('--full_flag', type=bool, default=False)  # full or reduced

    args = parser.parse_args()
    arg_str = [(str(key), str(value)) for (key, value) in vars(args).items()]
    print(arg_str)
    return args


def prepare_output_file(output=None, clean_flag=False, write_flag=False, data_folder='data/',
                        filename='minority_class_experiments_output.csv'):
    fieldnames = ["Model_Name", "Train_Acc", "Train_Loss",
                  "Valid_Acc", "Valid_Loss",
                  'BPM_Epoch']
    filename = data_folder + filename

    if clean_flag:
        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    if write_flag:
        if output is None:
            raise ValueError("Please specify output to write to output file.")
        with open(filename, 'a') as csvfile:
            print(output)
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(output)


def prepare_data(full_flag=False, minority_class=3, minority_percentage=0.01):
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
    if not full_flag:
        percentages[minority_class] = minority_percentage

    train_set = CIFAR10(root='data', transform=get_transform('train'), download=True, set_name='train',
                        percentages_list=percentages)

    if not full_flag:
        train_data = MinorityDataLoader(train_set, batch_size=64, shuffle=True, num_workers=4,
                                        minority_class_idx=minority_class)
    else:
        train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)

    return train_data, valid_data, test_data


def get_label_mapping():
    d = unpickle('data/cifar-10-batches-py/batches.meta')
    labels = d[b'label_names']
    return {value.decode('ascii'): index for index, value in enumerate(labels)}


def experiment(train, valid, num_epochs, title):
    model_save_dir = os.path.join(globals.ROOT_DIR, 'saved_models/' + title)
    train_results_file = os.path.join(globals.ROOT_DIR, 'results/' + title + '_train.txt')
    valid_results_file = os.path.join(globals.ROOT_DIR, 'results/' + title + '_valid.txt')

    model = DenseNet121()
    model = model.to(model.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                                momentum=MOMENTUM,
                                nesterov=True,
                                weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0001)
    train = (train, train_results_file)
    valid = (valid, valid_results_file)
    bpm = model.train_and_evaluate(num_epochs, optimizer, model_save_dir, train, scheduler, valid)
    return {"Model_Name": model_title, "Train_Acc": bpm['train_acc'], "Train_Loss": bpm['train_loss'],
              "Valid_Acc": bpm['valid_acc'], "Valid_Loss": bpm['valid_loss'],
              'BPM_Epoch': bpm['epoch']}


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

    # TRUE WHEN STARTING COMPLETELY NEW EXPERIMENT
    prepare_output_file(clean_flag=True)
    train_data, valid_data, test_data = prepare_data(full_flag=args.full_flag)

    # EXPERIMENT
    output = experiment(train_data, valid_data, args.num_epochs, model_title)

    prepare_output_file(clean_flag=False, write_flag=True, output=output)
