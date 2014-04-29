#!/bin/bash
# Torque submission script for SciNet gravity
#
# N.B: PBS lines are interpreted by qsub.  Change these defaults as 
# required
#
#PBS -l nodes=1:ppn=12:gpus=2,walltime=12:00:00
#PBS -N StackedAutoencoder
#PBS -q gravity

# Load theano modules
#cd ~/jobscripts
#sh load_theano_modules.sh

# Run the job

# To make substitutions from a higher up script: -a $FIRSTMODEL -b $SECONDMODEL -o $OFFSET -d $OUTDIRSUFFIX -l $LAYERTYPES
cd $PBS_O_WORKDIR
python ../pretrain_SdA_multiproc.py -d "${SCRATCH}/gpu_tests/SdA_results/5_layers/pretrain_output/$OUTDIRSUFFIX" -c 0.10 -a $FIRSTMODEL -b $SECONDMODEL -i "${SCRATCH}/sm_rep1_data/sm_rep1_screen.h5" -o $OFFSET -t $LAYERTYPES -l $LOSS



