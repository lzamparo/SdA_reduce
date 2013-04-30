#!/bin/bash
# Torque submission script for SciNet GPC
#
# N.B: PBS lines are interpreted by qsub.  Change these defaults as 
# required
#
#PBS -l nodes=1:ppn=8,walltime=4:00:00
#PBS -N plot_pli

# Load theano modules
#cd ~/jobscripts
#sh load_theano_modules.sh

# Run the job
cd $PBS_O_WORKDIR
python lle_embed_test.py --h5file "${SCRATCH}/sm_rep1_data/sample.h5" --size 3 --high 50 --low 10 --step 10 --iters 5 --output "${SCRATCH}/figures/letdata" 
