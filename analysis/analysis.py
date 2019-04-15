import csv
import numpy as np


def load_data(filename):
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
            if key not in data:
                data[key] = [(entry, seed)]
            else:
                add_flag = True
                for (existing_entry, existing_seed) in data[key]:
                    if existing_seed == seed:
                        add_flag = False
                if add_flag:
                    data[key].append((entry, seed))

    unfinished = []
    for key, value in data.items():
        if len(value) < 3:
            seeds = set([26, 27, 28])
            for (entry, seed) in value:
                seeds.remove(int(seed))
            unfinished.append((key, seeds))

    print(unfinished)
    return data


def process_data(data):
    entries = []
    #model_title -> {key: [values]}
    for key, values in data.items():
        processed_entry = {}
        output_entry = {}
        #create list of values
        for (entry, seed) in values:
            for key2, value2 in entry.items():
                if key2 == 'model_title':
                    processed_entry[key2] = value2
                    continue
                if key not in processed_entry:
                    processed_entry[key2] = [float(value2)]
                else:
                    processed_entry[key2].append(float(value2))

        #process list of values
        for key2, value2 in processed_entry.items():
            if key2 == 'model_title':
                output_entry[key2] = value2
            else:
                output_entry[key2 + '_std'] = float(np.std(value2))
                output_entry[key2 + '_mean'] = float(np.mean(value2))
        entries.append(output_entry)
    return entries

def write_data(entries, filename):
    fieldnames = list(entries[0].keys())
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry)

data = load_data('data/minority_class_experiments_overall.csv')
entries = process_data(data)
write_data(entries, 'data/processed_experiments_overall.csv')





