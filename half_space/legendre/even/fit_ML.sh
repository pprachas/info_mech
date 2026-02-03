#!/bin/bash -l

#$ -P lejlab2       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00  # Specify the hard time limit for the job
#$ -N fit_ML  # Give job a name
#$ -j y              # Merge the error and output streams into a single file
#$ -m ea

module load miniconda
conda activate fenicsx

python3 fit_legendre.py 4