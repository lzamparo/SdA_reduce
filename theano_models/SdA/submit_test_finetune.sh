#!/bin/bash
# Torque submission script for SciNet gravity
#
# N.B: PBS lines are interpreted by qsub.  Change these defaults as 
# required
#
#PBS -l nodes=1:ppn=12:gpus=2,walltime=1:00:00
#PBS -N TestFineTuneSdA
#PBS -q gravity

# Load theano modules
#cd ~/jobscripts
#sh load_theano_modules.sh

# Run the job

# To make substitutions from a higher up script: -p $FIRSTMODEL -q $SECONDMODEL -o $OFFSET
cd $PBS_O_WORKDIR
FIRSTMODEL='SdA_600_300_10.pkl'
SECONDMODEL='SdA_700_300_10.pkl'
python test_finetune_SdA.py -d "${SCRATCH}/gpu_tests/SdA_results/3_layers/finetune_output" -e "10/relu" -x "10/relu" -p $FIRSTMODEL -q $SECONDMODEL -i "${SCRATCH}/sm_rep1_data/sm_rep1_screen.h5" -o 5 -n 50.0



