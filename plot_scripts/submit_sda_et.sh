#!/bin/bash
# Torque submission script for SciNet GPC
#
# N.B: PBS lines are interpreted by qsub.  Change these defaults as 
# required
#
#PBS -l nodes=1:ppn=8,walltime=2:00:00
#PBS -N plot_plis

# Run the job
# Passed from top level dir: reducedbasedir, reducedfile, outputdir
cd $PBS_O_WORKDIR
python sda_embed_test.py --h5file "${SCRATCH}/sm_rep1_data/sample.h5" --reducedbasedir $BASEDIR --reducedfile $REDUCEDFILE --size 3 --iters 5 --outputdir $OUTPUTDIR 
