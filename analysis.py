import csv
import numpy as np


def load_data(filename):
    data = {}
    with open(filename) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for row in csv_reader:
            target = row['Target Percentage (in %)']
            label = row['Label']
            # seed = row['Seed']
            # num_epochs = row['Num Epochs']
            train_loss_full = float(row['Train_Loss_Full'])
            train_loss_mod = float(row['Train_Loss_Mod'])
            train_acc_mod = float(row['Train_Acc_Full'])
            train_acc_full = float(row['Train_Acc_Mod'])

            valid_acc_full = float(row['Valid_Acc_Full'])
            valid_acc_mod = float(row['Valid_Acc_Mod'])
            valid_loss_mod = float(row['Valid_Loss_Mod'])
            valid_loss_full = float(row['Valid_Loss_Full'])

            key = target + '_' + label
            if key not in data:
                data[key] = {'train_loss_full': [train_loss_full], 'valid_loss_mod': [valid_loss_mod],
                             'train_loss_mod': [train_loss_mod], 'train_acc_mod': [train_acc_mod],
                             'train_acc_full': [train_acc_full], 'valid_acc_full': [valid_acc_full],
                             'valid_acc_mod': [valid_acc_mod], 'valid_loss_full': [valid_loss_full]}
            else:
                data[key]['train_loss_full'].append(train_loss_full)
                data[key]['valid_loss_mod'].append(valid_loss_mod)
                data[key]['train_acc_mod'].append(train_acc_mod)
                data[key]['train_acc_full'].append(train_acc_full)
                data[key]['valid_acc_full'].append(valid_acc_full)
                data[key]['valid_acc_mod'].append(valid_acc_full)
                data[key]['valid_loss_full'].append(valid_loss_full)
                data[key]['train_loss_mod'].append(train_loss_mod)

    return data


def process_data(data):
    entries = []
    for key, value in data.items():
        target, label = key.split('_')
        train_loss_full_std = np.std(data[key]['train_loss_full'])
        train_loss_full_mean = np.mean(data[key]['train_loss_full'])

        train_acc_mod_std = np.std(data[key]['train_acc_mod'])
        train_acc_mod_mean = np.mean(data[key]['train_acc_mod'])

        train_acc_full_std = np.std(data[key]['train_acc_full'])
        train_acc_full_mean = np.mean(data[key]['train_acc_full'])

        train_loss_mod_std = np.std(data[key]['train_loss_mod'])
        train_loss_mod_mean = np.mean(data[key]['train_loss_mod'])

        valid_acc_full_std = np.std(data[key]['valid_acc_full'])
        valid_acc_full_mean = np.mean(data[key]['valid_acc_full'])

        valid_acc_mod_std = np.std(data[key]['valid_acc_mod'])
        valid_acc_mod_mean = np.mean(data[key]['valid_acc_mod'])

        valid_loss_mod_std = np.std(data[key]['valid_loss_mod'])
        valid_loss_mod_mean = np.mean(data[key]['valid_loss_mod'])

        valid_loss_full_std = np.std(data[key]['valid_loss_full'])
        valid_loss_full_mean = np.mean(data[key]['valid_loss_full'])

        entries.append([label, target,
                        round(train_loss_full_mean, 4), round(train_loss_full_std, 4),
                        round(train_loss_mod_mean, 4), round(train_loss_mod_std, 4),
                        round(valid_loss_full_mean, 4), round(valid_loss_full_std, 4),
                        round(valid_loss_mod_mean, 4), round(valid_loss_mod_std, 4),
                        round(train_acc_full_mean, 4), round(train_acc_full_std, 4),
                        round(train_acc_mod_mean, 4), round(train_acc_mod_std, 4),
                        round(valid_acc_full_mean, 4), round(valid_acc_full_std, 4),
                        round(valid_acc_mod_mean, 4), round(valid_acc_mod_std, 4)
                        ])
    return entries


def write_data(entries, filename):
    fieldnames = ['Label', 'Target (in %)',
                  'Train Loss Full Mean', 'Train Loss Full STD',
                  'Train Loss Mod Mean', 'Train Loss Mod STD',
                  'Valid Loss Full Mean', 'Valid Loss Full STD',
                  'Valid Loss Mod Mean', 'Valid Loss Mod STD',
                  'Train Acc Full Mean', 'Train Acc Full STD',
                  'Train Acc Mod Mean', 'Train Acc Mod STD',
                  'Valid Acc Full Mean', 'Valid Acc Full STD',
                  'Valid Acc Mod Mean', 'Valid Acc Mod STD',
                  ]

    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        for row in entries:
            writer.writerow(row)


data = load_data('data/minority_classes_output.csv')
entries = process_data(data)
write_data(entries, 'data/processed_output.csv')





