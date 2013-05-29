#!/bin/bash
# Torque submission script for SciNet gravity
#
# N.B: PBS lines are interpreted by qsub.  Change these defaults as 
# required
#
#PBS -l nodes=1:ppn=12:gpus=2,walltime=14:00:00
#PBS -N StackedAutoencoder
#PBS -q gravity

# Load theano modules
#cd ~/jobscripts
#sh load_theano_modules.sh

# Run the job
cd $PBS_O_WORKDIR
python test_multiproc_gpus.py -d "${SCRATCH}/gpu_tests/SdA_results" -c 0.10 -a 900-500-200-50 -b 1000-600-200-50 -i "${SCRATCH}/sm_rep1_data/sm_rep1_screen.h5" -o 3



