#!/bin/bash

#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --job-name=era5_pic_generation
#SBATCH --output=era5_pic_generation.out
#SBATCH --error=era5_pic_generation.err

module load anaconda/2021.11
source activate /p/tmp/bochow/lama_env/
export TORCH_HOME=/p/tmp/bochow/LAMA/lama/ && export PYTHONPATH=/p/tmp/bochow/LAMA/lama/
module load cuda/10.2
export HDF5_USE_FILE_LOCKING='FALSE'


srun --ntasks=1 --cpus-per-task=1  python /p/tmp/bochow/LAMA/lama/era5-dataset/generate_images.py
