#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=100:00:00
#SBATCH --mem=80000
#SBATCH --job-name=gpu
#SBATCH --output=hadcrut_cr.out
#SBATCH --error=hadcrut_cr.err

module load anaconda/2021.11
source activate /p/tmp/bochow/lama_env/
export TORCH_HOME=/p/tmp/bochow/LAMA/lama/ && export PYTHONPATH=/p/tmp/bochow/LAMA/lama/
module load cuda/10.2
export HDF5_USE_FILE_LOCKING='FALSE'


srun --ntasks=1 --cpus-per-task=16 python bin/train.py -cn lama-fourier-hadcrut data.batch_size=90


