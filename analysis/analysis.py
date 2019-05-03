import csv
import numpy as np


def load_data(filename):
    """
    :param filename: reading 120 experiments from
    :return: data experiment -> exp_data
    """
    data = {}
    with open(filename) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for row in csv_reader:
            entry = {}
            for header in csv_reader.fieldnames:
                entry[header] = row[header]

            key = entry['model_title'][:-3]
            seed = entry['model_title'][-2:]
            entry['model_title'] = key
            print(row)
            print(key, seed)
            if seed == '96':
                exit()
            if key not in data:
                data[key] = [(entry, seed)]
            else:
                add_flag = True
                for (existing_entry, existing_seed) in data[key]:
                    if existing_seed == seed:
                        add_flag = False
                if add_flag:
                    data[key].append((entry, seed))

    # Were any experiments unfinished?
    unfinished = []
    for key, value in data.items():
        if len(value) < 3:
            print(key)
            seeds = set([26, 27, 28])
            for (entry, seed) in value:
                seeds.remove(int(seed))
            unfinished.append((key, seeds))

    return data


def process_data(data):
    entries = []
    mappings = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    #model_title -> [seed values]
    for key, values in data.items():
        processed_entry = {}
        output_entry = {}

        # create list of values
        for (entry, seed) in values:
            for key2, value2 in entry.items():
                if key2 == 'title':
                    continue
                # filter table
                if 'loss' in key2 or 'valid' in key2 or 'epoch' in key2 or 'minority' in key2:
                    continue
                if len(key2.split('_')) > 3:
                    class_id = key2.split('_')[-1]
                    key2 = '_'.join(key2.split('_')[:-1]) + '_' + mappings[int(class_id)]
                if key2 == 'model_title':
                    processed_entry[key2] = value2
                    continue

                if key2 not in processed_entry:
                    processed_entry[key2] = [float(value2)]
                else:
                    processed_entry[key2].append(float(value2))
                    processed_entry[key2].append(float(value2))

        #process list of values
        for key2, value2 in processed_entry.items():
            if key2 == 'model_title':
                output_entry[key2] = value2
            else:
                mean = round(float(np.mean(value2) * 100), 2)
                std = round(float(np.std(value2) * 100), 2)
                output_entry[key2] = str(mean) + '% ± ' + str(std) + '%'

        entries.append(output_entry)
    return sorted(entries, key=lambda k: k['model_title'])


def write_data(entries, filename):
    fieldnames = sorted(list(entries[0].keys()))

    # to_print = [key.replace('_', ' ').upper() for key in fieldnames]
    # print(' & '.join(to_print))

    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            output = entry['model_title']
            for key in fieldnames:
                if key == 'model_title':
                    continue
                value = entry[key].replace('±', '$\pm$')
                value = value.replace('%', '\%')
                color_cell = ''
                if entry['model_title'].split('_')[0] in key:
                    color_cell = '\cellcolor{LightCyan} '

                output += ' & ' + color_cell + value

            print(output)
            writer.writerow(entry)


for key in ['overall.csv', 'minority.csv']:
    filename = 'data/minority_class_experiments_bpm_' + key
    output_filename = 'data/processed_experiments_bpm_' + key
    data = load_data(filename)
    entries = process_data(data)
    write_data(entries, output_filename)





