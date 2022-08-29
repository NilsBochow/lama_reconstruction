#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=48:00:00
#SBATCH --mem=60000
#SBATCH --job-name=gpu
#SBATCH --output=gpu.out
#SBATCH --error=gpu.err

module load anaconda/2021.11
source activate /p/tmp/bochow/lama_env/
export TORCH_HOME=/p/tmp/bochow/LAMA/lama/ && export PYTHONPATH=/p/tmp/bochow/LAMA/lama/
module load cuda/10.2
export HDF5_USE_FILE_LOCKING='FALSE'


python3 bin/train.py -cn lama-fourier-celeba data.batch_size=10


