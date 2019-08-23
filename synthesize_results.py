import argparse
import os
from globals import ROOT_DIR
import csv
import numpy as np


RESERVED_DIRECTORIES = ['comet', 'saved_models']


def num(value):
    try:
        return float(value)
    except ValueError:
        return value


def prepare_output_file(filename, output=None, file_action_key='a+', aggregate=False):
    """
    :param filename:
    :param output: dictionary to write to csv
    :param clean_flag: bool to delete existing dictionary
    :param file_action_key: w to write or a+ to append to file
    :return:
    """
    file_exists = os.path.isfile(filename)

    if output is None or output == []:
        raise ValueError("Please specify output list to write to output file.")
    with open(filename, file_action_key) as csvfile:
        fieldnames = sorted(list(output[0].keys())) # to make sure new dictionaries in diff order work okay
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists or file_action_key == 'w' or os.path.getsize(filename) == 0:
            writer.writeheader()

        for entry in output:
            writer.writerow(entry)


def aggregate_test_results(path):
    if not os.path.isfile(path):
        raise ValueError("File {} does not exist".format(path))
    data = {}
    print("Path {}".format(path))
    # organize by last 3 recorded seeds
    with open(path) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            entry = {key: row[key] for key in csv_reader.fieldnames}
            data[row['seed']] = entry

    processed_data = {}
    for _, entry in data.items():
        for key, value in entry.items():

            value = num(value)
            if value == '-':
                processed_data[key] = '-'
            elif isinstance(value, str):
                processed_data[key] = value
            elif key not in processed_data:
                processed_data[key] = [value]
            else:
                processed_data[key].append(value)
    output = {}
    for key, value in processed_data.items():
        if key == 'seed':
            output['num_experiments'] = len(value)
        elif key == 'epoch' and value != '-':
            output['epoch'] = int(value[0])
        elif isinstance(value[0], float):
            output[key] = '{} ± {}'.format(np.around(np.mean(value), 4), np.around(np.std(value), 4))
        else:
            output[key] = value
    return output


def save_existing_experiments(path):
    data = {}
    if os.path.isfile(path):
        with open(path) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                entry = {key: row[key] for key in csv_reader.fieldnames}
                data[row['title']] = entry
    print("Data from existing experiments: {}".format(data))
    return data


def fetch_directories(results_dir):
    directories = []
    dir_list_list = [dir_names for (_, dir_names, _) in os.walk(results_dir)]
    for dir_list in dir_list_list:
        for dir in dir_list:
            if 'test' in dir:
                continue
            directories.append(dir)
    return directories


def fetch_files(results_dir):
    files = []
    filename_list_list = [file_names for (_, _, file_names) in os.walk(results_dir)]
    for filename_list in filename_list_list:
        for file in filename_list:
            files.append(file)
    return [file for file in files if file[-4:] == '.zip']


def upload_to_comet(results_dir):
    output = fetch_files(results_dir)
    comet_files_path = os.path.join(results_dir, 'comet_files.txt')
    lines = set()
    if os.path.isfile(comet_files_path):
        with open(comet_files_path, 'r') as f:
            lines = set(f.readlines())
    for comet_file in output:
        command = "comet upload {}/{}\n".format(results_dir, comet_file)
        if command not in lines:
            print("Comet file with command {} added".format(command))
            os.system(command)
            lines.add(command)
    with open(comet_files_path, 'w') as f:
        for line in lines:
            f.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Synthesizing Experiment Data...')
    parser.add_argument('--upload_comet', type=bool, default=False)
    args = parser.parse_args()

    # for each folder in results
    test_results_filename = 'results.csv'
    results_dir = os.path.join(ROOT_DIR, 'results/')
    experiments_results_filename = os.path.join(results_dir, 'experiments.csv')
    # experiment_data = save_existing_experiments(experiments_results_filename)
    experiment_data = {}
    print("=== Synthesizing Global Experiment Results ===")
    directories = fetch_directories(results_dir)
    for directory in directories:
        if directory in RESERVED_DIRECTORIES:
            continue
        if args.upload_comet:
            upload_to_comet(os.path.join(results_dir, directory))

        output = aggregate_test_results(os.path.join(results_dir, '{}/{}'.format(directory, test_results_filename)))
        experiment_data[output['title']] = output

    print("Experiment entries are: {}".format(list(experiment_data.keys())))

    if len(experiment_data) > 0:
        print("Saving to file {}".format(experiments_results_filename))
        prepare_output_file(filename=experiments_results_filename,
                            output=list(experiment_data.values()),
                            file_action_key='w',
                            aggregate=True)
