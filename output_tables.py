"""
Creates Latex table for paper
"""
import os
from globals import ROOT_DIR
import csv
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Output tables...')
    parser.add_argument('--large', type=bool, default=True)
    args = parser.parse_args()
    arg_str = [(str(key), str(value)) for (key, value) in vars(args).items()]
    print("Arguments: {}".format(arg_str))
    return args


def large_table():
    data = []
    path = os.path.join(ROOT_DIR, 'results/experiments.csv')
    with open(path) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # entry = {key: row[key] for key in csv_reader.fieldnames if 'test' in key}
            target_class, size = row['title'].split('_')
            f_score, f_score_class = row['test_f_score'], row['test_f_score_{}'.format(target_class)]
            size = float(size) * 100
            data.append([target_class, str(int(size)) + '\%', f_score, f_score_class])
    data = sorted(data, key=lambda x: (x[0], x[1]))

    for i, item in enumerate(data):
        if i == len(data) - 1:
            print("\\belowspace")
        if item[0] == 'cat':
            print("\\rowcolor{LightCyan}")
        if i % 3 == 0 and i != 0:
            print("\\hline")
        output_item = ' & '.join(item)
        output_item = output_item.replace('±', '$\pm$')
        print("{} \\\\".format(output_item))


# create table with label, class size, test set f-score, test set target class f-score
if __name__ == "__main__":
    args = get_args()
    if args.large:
        large_table()
