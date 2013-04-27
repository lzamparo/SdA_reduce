#!/bin/bash
# Torque submission script for SciNet ARC
#
# N.B: PBS lines are interpreted by qsub.  Change these defaults as 
# required
#
#PBS -l nodes=1:ppn=8:gpus=1,walltime=3:00:00
#PBS -N GPUtest
#PBS -q arc

# Load theano modules
#cd ~/jobscripts
#sh load_theano_modules.sh

# Run the job
cd $PBS_O_WORKDIR
python drive_dA.py -d "${SCRATCH}/gpu_tests/dA_results" -c 0.30
