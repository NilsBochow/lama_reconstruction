#!/bin/bash


srun --ntasks=1 --cpus-per-task=1 --time=1:00:00 --qos=priority python /p/tmp/bochow/LAMA/lama/hadcrut/plot_inference.py 
