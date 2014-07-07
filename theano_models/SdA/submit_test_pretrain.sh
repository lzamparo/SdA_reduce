#!/bin/bash
# Torque submission script for SciNet gravity
#
# N.B: PBS lines are interpreted by qsub.  Change these defaults as 
# required
#
#PBS -l nodes=1:ppn=12:gpus=2,walltime=04:00:00
#PBS -N Test_Hybrid_Pretrain
#PBS -q gravity

# Load theano modules
#cd ~/jobscripts
#sh load_theano_modules.sh

# Run the job

# To make substitutions from a higher up script: -p $FIRSTMODEL -q $SECONDMODEL -o $OFFSET
cd $PBS_O_WORKDIR
python test_pretrain_SdA.py -d /scratch/z/zhaolei/lzamparo/gpu_tests/ -i "${SCRATCH}/sm_rep1_data/sm_rep1_screen.h5" -c 0.25 -s $SCRATCH/gpu_tests/adaptive_lr_pretrain.pkl -r $SCRATCH/gpu_tests/adaptive_lr_pretrain.pkl



