#!/bin/bash -l
# Batch script to run a serial job on Legion with the upgraded
# software stack under SGE.

# 1. Force bash as the executing shell.
#$ -S /bin/bash


# 3. Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=8:00:0

# 4. Request 4 gigabyte of RAM (must be an integer)
#$ -l mem=16G

# 5. Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=20G

# 6. Select the MPI parallel environment and 10 processes.
#$ -pe mpi 10

# 7. Set the name of the job.
#$ -N training_CNN_new_dataset

# 8. Set the working directory to somewhere in your scratch space.  This is
# a necessary step with the upgraded software stack as compute nodes cannot
# write to $HOME.
#$ -wd /home/ucesxc0/Scratch/output/training_CNN_new_dataset

# 9. load the cuda module (in case you are running a CUDA program
module load python3/recommended
module load tensorflow/1.12.0/cpu
./training_network.py
