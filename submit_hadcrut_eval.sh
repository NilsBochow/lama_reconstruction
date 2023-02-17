#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --qos=gpushort
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=0:10:00
#SBATCH --mem=40000
#SBATCH --job-name=gpu
#SBATCH --output=hadcrut_eval.out
#SBATCH --error=hadcrut_eval.err

module load anaconda/2021.11
source activate /p/tmp/bochow/lama_env/
export TORCH_HOME=/p/tmp/bochow/LAMA/lama/ && export PYTHONPATH=/p/tmp/bochow/LAMA/lama/
module load cuda/10.2
export HDF5_USE_FILE_LOCKING='FALSE'


#srun --ntasks=1 --cpus-per-task=4 --time=0:10:00 --qos=priority python bin/predict.py model.path=$(pwd)/experiments/bochow_2022-08-18_11-37-10_train_lama-fourier-hadcrut_/ indir=$(pwd)/hadcrut/eval/fixed_72.yaml/ outdir=$(pwd)/inference/hadcrut/fixed_72 model.checkpoint=last.ckpt


#srun --ntasks=1 --cpus-per-task=4 --time=0:10:00 --qos=priority python bin/evaluate_predicts.py $(pwd)/configs/eval2_gpu.yaml $(pwd)/hadcrut/eval/fixed_72.yaml/ $(pwd)/inference/hadcrut/fixed_72 $(pwd)/inference/hadcrut/fixed_72_metrics.csv

#srun --ntasks=1 --cpus-per-task=4 --time=0:10:00 --qos=priority python bin/predict.py model.path=$(pwd)/experiments/bochow_2022-09-13_11-07-53_train_lama-fourier-hadcrut_/ indir=$(pwd)/hadcrut/hadcrut_missing_masks/fixed_72.yaml/ outdir=$(pwd)/inference/hadcrut/fixed_72_2 model.checkpoint=last.ckpt

#srun --ntasks=1 --cpus-per-task=4 --time=0:10:00 --qos=priority python bin/predict.py model.path=$(pwd)/experiments/bochow_2022-09-21_10-17-11_train_lama-fourier-hadcrut_/ indir=$(pwd)/hadcrut/hadcrut_missing_masks/fixed_72.yaml/ outdir=$(pwd)/inference/hadcrut/fixed_72_10days model.checkpoint=epoch=46-step=317561.ckpt

srun --ntasks=1 --cpus-per-task=4 --time=0:10:00 --qos=priority python bin/predict.py model.path=$(pwd)/experiments/bochow_2023-02-10_09-23-51_train_lama-fourier-sic_ indir=/p/tmp/bochow/sic_era5/Niklas_1440/fixed_1440.yaml/ outdir=/p/tmp/bochow/sic_era5/Niklas_1440/inpainted/ model.checkpoint=last.ckpt

