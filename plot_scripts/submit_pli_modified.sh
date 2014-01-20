#!/bin/bash
# Torque submission script for SciNet GPC
#
# N.B: PBS lines are interpreted by qsub.  Change these defaults as 
# required
#
#PBS -l nodes=1:ppn=8,walltime=3:00:00
#PBS -N plot_pli

# Load theano modules
#cd ~/jobscripts
#sh load_theano_modules.sh

# Run the job
cd $PBS_O_WORKDIR
python plot_pca_lle_modified_isomap.py --h5file "${SCRATCH}/sm_rep1_data/sample.h5" --size 7 --sample-size 3000 --dimension 2 --output "${SCRATCH}/figures/pli_modified.eps" 
