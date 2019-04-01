#!/bin/bash -l
# Batch script to run a serial job on Legion with the upgraded
# software stack under SGE.
# 1. Force bash as the executing shell.
#$ -S /bin/bash

# 2. Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=4:0:0

# 3. Request 16 gigabyte of RAM (must be an integer)
#$ -l mem=16G

# 4. Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=15G

# 5. Set the name of the job.
#$ -N image_trajectory_generation

#6. Select the MPI parallel environment and 16 processes.
#$ -pe mpi 16

# 7. Set the working directory to somewhere in your scratch space. This is
# a necessary step with the upgraded software stack as compute nodes cannot
# write to $HOME.
#$ -wd /home/ucesxc0/Scratch/output/process_ais_data_Danish

#8. run the application
module load python3/recommended
./trajectory_image_generation.py