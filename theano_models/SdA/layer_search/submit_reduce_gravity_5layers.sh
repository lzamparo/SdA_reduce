#!/bin/bash
# Torque submission script for SciNet gravity
#
# N.B: PBS lines are interpreted by qsub.  Change these defaults as 
# required
#
#PBS -l nodes=1:ppn=12:gpus=2,walltime=3:00:00
#PBS -N ReduceStackedAutoencoder
#PBS -q gravity

# Run the job

# To make substitutions from a higher up script: -p $FIRSTMODEL -q $SECONDMODEL -o $OFFSET -x $EXTENSION
cd $PBS_O_WORKDIR
python reduce_SdA_multiproc.py -d "${SCRATCH}/gpu_tests/SdA_results/reduced_data/5_layers" -x $EXTENSION -p $FIRSTMODEL -q $SECONDMODEL -i "${SCRATCH}/sm_rep1_data/sample.h5" 



