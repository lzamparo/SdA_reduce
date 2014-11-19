#!/bin/bash
# Kernel PCA tuning script for GPC
#
# N.B: PBS lines are interpreted by qsub.  Change these defaults as 
# required
#
#PBS -l nodes=1:ppn=8,walltime=12:00:00
#PBS -N kpca_tuning

# Load theano modules
#cd ~/jobscripts
#sh load_theano_modules.sh

# Run the job
cd $PBS_O_WORKDIR
python kernel_pca_pipeline.py --h5file "${SCRATCH}/sm_rep1_data/sample.h5" --size 5 --sample-size 5000 --num-jobs 8 --output "${SCRATCH}/figures/kpca_pipeline.pkl" 
