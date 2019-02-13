import pickle
import os
import csv
from collections import OrderedDict

def save_statistics(statistics_to_save,file_path):
    '''
    :param statistics_to_save: dict, val type is float
    :param file_path: e.g. file_path = "C:/test_storage_utils/dir2/test.txt"
    '''
    if type(statistics_to_save) is not OrderedDict:
        raise TypeError('statistics_to_save must be OrderedDict instead got {}'.format(type(statistics_to_save)))

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path,'a+') as f: # append mode + creates if doesn't exist
        header = ""
        line = ""
        for i,key in enumerate(statistics_to_save.keys()):
            val = statistics_to_save[key]
            if i==0:
                line = line + "{:.4f}".format(val)
                header = header + key
            else:
                line = line + "\t" + "{:.4f}".format(val)
                header = header + "\t" + key
        if os.stat(file_path).st_size == 0:  # if empty
            f.write(header+"\n")
        f.write(line+"\n")

def main():
    pass

if __name__ == '__main__':
    main()