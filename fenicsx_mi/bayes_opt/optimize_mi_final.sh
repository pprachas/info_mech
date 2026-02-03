#!/bin/bash -l

#$ -P lejlab2       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00  # Specify the hard time limit for the job
#$ -N final_mi  # Give job a name
#$ -l mem_per_core=8G
#$ -j y              # Merge the error and output streams into a single file
#$ -m ea
#$ -t 1-2

module load miniconda
conda activate fenicsx

python3 optimize_final_mi.py $SGE_TASK_ID