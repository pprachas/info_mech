#!/bin/bash -l

#$ -P lejlab2       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00  # Specify the hard time limit for the job
#$ -N pointwise_sigma  # Give job a name
#$ -j y              # Merge the error and output streams into a single file
#$ -l mem_per_core=8G
#$ -m ea

module load miniconda/23.1.0
conda activate fenicsx

python3 pointwise_sigma.py