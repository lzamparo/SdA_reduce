#!/bin/bash
# Torque submission script for SciNet ARC
#
# N.B: PBS lines are interpreted by qsub.  Change these defaults as 
# required
#
#PBS -l nodes=1:ppn=8:gpus=1,walltime=5:00:00
#PBS -N dA_foci
#PBS -q arc

# Load theano modules
#cd ~/jobscripts
#sh load_theano_modules.sh

# Run the job
cd $PBS_O_WORKDIR
python test_pickle_sqe_dA.py -d "${SCRATCH}/gpu_tests/dA_results" -i "${SCRATCH}/sm_rep1_data/sample.h5" -c 0.20
