#!/bin/bash
# Torque submission script for SciNet gravity
#
# N.B: PBS lines are interpreted by qsub.  Change these defaults as 
# required
#
#PBS -l nodes=1:ppn=12:gpus=2,walltime=5:00:00
#PBS -N ReduceSamplePopulations
#PBS -q gravity

# load python modules
source ~/.bashrc_python2

# parameters
dir="${SCRATCH}/gpu_models/more_features/5_layers/finetune_pkl_files/10"
sda="SdA_1100_600_400_80_10.pkl"
inputfile="/scratch/z/zhaolei/lzamparo/sm_rep1_data/reference_samples/ref_pops.h5"
outputfile="/scratch/z/zhaolei/lzamparo/sm_rep1_data/reference_samples/ref_pops_topmodel_10.h5"

# To make substitutions from a higher up script: -p $FIRSTMODEL -q $SECONDMODEL -o $OFFSET -d $DIR -x $DIM

cd $PBS_O_WORKDIR
python reduce_SdA.py -d "$dir" -r "$sda" -i "$inputfile" -o "$outputfile"
