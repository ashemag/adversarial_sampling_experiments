"""
Dynamically generates and runs minority class experiment scripts
to test out combinations of labels, seeds, target percentages
of CIFAR-10. Will continue trying script on MLP cluster until it
successfully finishes
"""
import os
import shutil
import subprocess
import argparse
import tqdm
import getpass
import time

parser = argparse.ArgumentParser(description='Welcome to the run N at a time script')
parser.add_argument('--num_parallel_jobs', type=int)
parser.add_argument('--total_epochs', type=int)
args = parser.parse_args()

def check_if_experiment_with_name_is_running(experiment_name):
    result = subprocess.run(['squeue --name {}'.format(experiment_name), '-l'], stdout=subprocess.PIPE, shell=True)
    lines = result.stdout.split(b'\n')
    if len(lines) > 2:
        return True
    else:
        return False

student_id = getpass.getuser().encode()[:5]


def create_files(file_flag=False):
    # for each class, # for each seed value # for each target percentage
    cifar_classes = ['horse', 'frog', 'cat', 'airplane', 'automobile', 'ship', 'truck', 'deer', 'bird', 'dog']
    seed_values = [28, 27, 26]
    target_percentages = [-1, 10, 1, .1] #-1 == 100
    for cifar_class in cifar_classes:
        for seed in seed_values:
            for target_percentage in target_percentages:
                command = 'python minority_class_experiments.py --num_epochs 120 --label ' + cifar_class
                command += ' --seed ' + str(seed)
                script_name = str(cifar_class) + '_' + str(seed) + '_'
                if target_percentage == -1:
                    script_name += 'full.sh'
                    command += ' --full_flag True'
                else:
                    script_name += str(target_percentage) + '.sh'
                    command += ' --target_percentage ' + str(target_percentage)
                print(script_name)
                if file_flag:
                    with open(script_name, 'w') as f:
                        shutil.copy2('template.sh', script_name)

                    with open(script_name, 'a') as f:
                        f.write(command)
                    time.sleep(3) # t in seconds

create_files(file_flag=False)
list_of_scripts = [item for item in
                   subprocess.run(['ls'], stdout=subprocess.PIPE).stdout.split(b'\n') if
                   item.decode("utf-8").endswith(".sh")]


for script in list_of_scripts:
    print('sbatch', script.decode("utf-8"))

epoch_dict = {key.decode("utf-8"): 0 for key in list_of_scripts}
total_jobs_finished = 0

while total_jobs_finished < args.total_epochs * len(list_of_scripts):
    curr_idx = 0
    with tqdm.tqdm(total=len(list_of_scripts)) as pbar_experiment:
        while curr_idx < len(list_of_scripts):
            number_of_jobs = 0
            result = subprocess.run(['squeue', '-l'], stdout=subprocess.PIPE)
            for line in result.stdout.split(b'\n'):
                if student_id in line:
                    number_of_jobs += 1

            if number_of_jobs < args.num_parallel_jobs:
                while check_if_experiment_with_name_is_running(
                        experiment_name=list_of_scripts[curr_idx].decode("utf-8")) or epoch_dict[
                    list_of_scripts[curr_idx].decode("utf-8")] >= args.total_epochs:
                    curr_idx += 1
                    if curr_idx >= len(list_of_scripts):
                        curr_idx = 0

                str_to_run = 'sbatch {}'.format(list_of_scripts[curr_idx].decode("utf-8"))
                total_jobs_finished += 1
                os.system(str_to_run)
                print(str_to_run)
                curr_idx += 1
            else:
                time.sleep(1)
