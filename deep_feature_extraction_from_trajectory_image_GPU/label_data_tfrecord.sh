#!/usr/bin/env bash
#!/bin/bash -l
# Batch script to run a serial job on Legion with the upgraded
# software stack under SGE.
# 1. Force bash as the executing shell.
#$ -S /bin/bash

# 2. Request a number of GPU cards, in this case 2 (the maximum)
#$ -l gpu=1

# 3. Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=2:0:0

# 3. Request 16 gigabyte of RAM (must be an integer)
#$ -l mem=16G

# 4. Request 20 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=20G

# 5. Set the name of the job.
#$ -N label_image_to_tfrecord


# 6. Set the working directory to somewhere in your scratch space. This is
# a necessary step with the upgraded software stack as compute nodes cannot
# write to $HOME.
#$ -wd /home/ucesxc0/Scratch/output/training_CNN_new_dataset

# 7. run the application
module unload compilers mpi
module load compilers/gnu/4.9.2
module load python3/recommended
module load cuda/9.0.176-patch4/gnu-4.9.2
module load cudnn/7.4.2.24/cuda-9.0
module load tensorflow/1.12.0/gpu
./label_data_tfrecord.py