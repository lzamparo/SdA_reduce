#!/bin/bash
# Torque submission script for SciNet GPC
#
# N.B: PBS lines are interpreted by qsub.  Change these defaults as 
# required
#
#PBS -l nodes=1:ppn=8,walltime=2:00:00
#PBS -N plot_plis

# Run the job
cd $PBS_O_WORKDIR
python sda_embed_test.py --h5file "${SCRATCH}/sm_rep1_data/sample.h5" --reducedbasedir "${SCRATCH}/gpu_models/SdA/reduced_data/batch_1000_size_25ch_mom_0.8_wd_0.00001_lr_0.005/" --reducedfile $REDUCEDFILE --size 3 --iters 5 --outputdir "${SCRATCH}/figures/setdata" 
