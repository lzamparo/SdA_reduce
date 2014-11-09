
#!/bin/bash
# Torque submission script for SciNet gravity
#
# N.B: PBS lines are interpreted by qsub.  Change these defaults as
# required
#
#PBS -l nodes=1:ppn=12:gpus=2,walltime=6:00:00
#PBS -N FineTuneStackedAutoencoder
#PBS -q gravity

# Load modules 
source ~/.bashrc_python2

# Run the job

# To make substitutions from a higher up script: -p $FIRSTMODEL -q $SECONDMODEL -o $OFFSET
cd $PBS_O_WORKDIR
python finetune_SdA_multiproc.py -d "${SCRATCH}/gpu_models/more_features/5_layers/finetune_output" -e "10" -x "10" -p $FIRSTMODEL -q $SECONDMODEL -i "${SCRATCH}/sm_rep1_data/sm_rep1_screen.h5" -o $OFFSET -s "adagrad_momentum_wd"



