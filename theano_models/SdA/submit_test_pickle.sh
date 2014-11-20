#!/bin/bash
# Torque submission script for SciNet gravity
#
# N.B: PBS lines are interpreted by qsub.  Change these defaults as 
# required
#
#PBS -l nodes=1:ppn=12:gpus=2,walltime=00:30:00
#PBS -N Test_Pickle_SdA
#PBS -q gravity

# Load theano modules
source ~/.bashrc_python2

# Run the job

# To make substitutions from a higher up script: -p $FIRSTMODEL -q $SECONDMODEL -o $OFFSET
cd $PBS_O_WORKDIR
python test_pickle_SdA.py -d /scratch/z/zhaolei/lzamparo/gpu_tests/ -i "${SCRATCH}/sm_rep1_data/sm_rep1_screen.h5" -c 0.25 -s $SCRATCH/gpu_tests/sda_pickle.pkl -r $SCRATCH/gpu_tests/sda_pickle.pkl



