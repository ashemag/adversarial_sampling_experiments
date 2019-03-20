#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Standard
#SBATCH --gres=gpu:2
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/
# Activate the relevant virtual environment:


source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
cd ..

# dog, ship, truck, airplane, automobile, bird, cat, deer, frog, horse
# completed: truck, dog

python data_experiments.py --label "airplane" --seed 28 --num_epochs 120 --target_percentage 1 --train_data 'full'
#python data_experiments.py --label "truck" --seed 27 --num_epochs 120 --target_percentage 1 --train_data 'full'
#python data_experiments.py --label "truck" --seed 26 --num_epochs 120 --target_percentage 1 --train_data 'full'

#python data_experiments.py --label "truck" --seed 28 --num_epochs 120 --target_percentage .01 --train_data 'reduced'
#python data_experiments.py --label "truck" --seed 27 --num_epochs 120 --target_percentage .01 --train_data 'reduced'
#python data_experiments.py --label "truck" --seed 26 --num_epochs 120 --target_percentage .01 --train_data 'reduced'

#python data_experiments.py --label "truck" --seed 28 --num_epochs 120 --target_percentage .1 --train_data 'reduced'
#python data_experiments.py --label "truck" --seed 27 --num_epochs 120 --target_percentage .1 --train_data 'reduced'
#python data_experiments.py --label "truck" --seed 26 --num_epochs 120 --target_percentage .1 --train_data 'reduced'

#python data_experiments.py --label "truck" --seed 28 --num_epochs 120 --target_percentage 1 --train_data 'reduced'
#python data_experiments.py --label "truck" --seed 27 --num_epochs 120 --target_percentage 1 --train_data 'reduced'
#python data_experiments.py --label "truck" --seed 26 --num_epochs 120 --target_percentage 1 --train_data 'reduced'

"""

Watch outs

GPU memory allocation will fail within first 5 minutes sometimes. Make sure job runs past that
Make sure only 1 model is running at a time, otherwise will timeout
"""
