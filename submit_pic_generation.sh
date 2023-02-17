#!/bin/bash

#SBATCH --qos=priority
#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --ntasks=12
#SBATCH --job-name=era5_pic_generation
#SBATCH --output=era5_pic_generation.out
#SBATCH --error=era5_pic_generation.err

module load anaconda/2021.11
source activate /p/tmp/bochow/lama_env/
export TORCH_HOME=/p/tmp/bochow/LAMA/lama/ && export PYTHONPATH=/p/tmp/bochow/LAMA/lama/
module load cuda/10.2
export HDF5_USE_FILE_LOCKING='FALSE'


##srun --ntasks=1 --cpus-per-task=1  python /p/tmp/bochow/LAMA/lama/era5-dataset/generate_images.py


## HADCRUT
#srun --ntasks=1 --cpus-per-task=4 --time=0:10:00 --qos=priority python bin/gen_mask_dataset.py $(pwd)/configs/data_gen/fixed_72.yaml hadcrut/val_source/ hadcrut/val/fixed_72.yaml  --ext png &

#srun --ntasks=1 --cpus-per-task=4 --time=0:10:00 --qos=priority python bin/gen_mask_dataset.py $(pwd)/configs/data_gen/fixed_72.yaml hadcrut/eval_source/ hadcrut/eval/fixed_72.yaml  --ext png &

#srun --ntasks=1 --cpus-per-task=4 --time=0:10:00 --qos=priority python bin/gen_mask_dataset.py $(pwd)/configs/data_gen/fixed_72.yaml hadcrut/visual_test_source/ hadcrut/visual_test/fixed_72.yaml  --ext png &


#srun --ntasks=1 --cpus-per-task=4 --time=0:10:00 --qos=priority python bin/gen_mask_dataset.py $(pwd)/configs/data_gen/fixed_256.yaml era5-dataset/val_source/ era5-dataset/val/fixed_256.yaml  --ext png &

#srun --ntasks=1 --cpus-per-task=4 --time=0:10:00 --qos=priority python bin/gen_mask_dataset.py $(pwd)/configs/data_gen/fixed_256.yaml era5-dataset/eval_source/ era5-dataset/eval/fixed_256.yaml  --ext png &

#srun --ntasks=1 --cpus-per-task=4 --time=0:10:00 --qos=priority python bin/gen_mask_dataset.py $(pwd)/configs/data_gen/fixed_256.yaml era5-dataset/visual_test_source/ era5-dataset/visual_test/fixed_256.yaml  --ext png &

#srun --ntasks=1 --cpus-per-task=4 --time=0:10:00 --qos=priority python bin/gen_mask_dataset.py $(pwd)/configs/data_gen/fixed_1440.yaml /p/tmp/bochow/sic_era5/val_source_orig/ /p/tmp/bochow/sic_era5/val_orig/fixed_1440.yaml  --ext png &
#srun --ntasks=1 --cpus-per-task=4 --time=0:10:00 --qos=priority python bin/gen_mask_dataset.py $(pwd)/configs/data_gen/fixed_1440.yaml /p/tmp/bochow/sic_era5/eval_source_orig/ /p/tmp/bochow/sic_era5/eval_orig/fixed_1440.yaml  --ext png &
#srun --ntasks=1 --cpus-per-task=4 --time=0:10:00 --qos=priority python bin/gen_mask_dataset.py $(pwd)/configs/data_gen/fixed_1440.yaml /p/tmp/bochow/sic_era5/visual_test_source_orig/ /p/tmp/bochow/sic_era5/visual_test_orig/fixed_1440.yaml  --ext png &

srun --ntasks=1 --cpus-per-task=4 --time=0:10:00 --qos=priority python bin/gen_mask_dataset.py $(pwd)/configs/data_gen/fixed_1440.yaml /p/tmp/bochow/sic_era5/Niklas_1440/ /p/tmp/bochow/sic_era5/Niklas_1440/fixed_1440.yaml  --ext png &
wait
