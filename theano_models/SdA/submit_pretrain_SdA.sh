#!/bin/bash
# Torque submission script for SciNet ARC
#
# N.B: PBS lines are interpreted by qsub.  Change these defaults as 
# required
#
#PBS -l nodes=1:ppn=8:gpus=1,walltime=10:00:00
#PBS -N StackedAutoencoder
#PBS -q arc

# Load theano modules
#cd ~/jobscripts
#sh load_theano_modules.sh

# Run the job
cd $PBS_O_WORKDIR
python pretrain_SdA.py -d "${SCRATCH}/gpu_tests/SdA_results" -c 0.10 -s SdA_850_400_50.pkl -i "${SCRATCH}/sm_rep1_data/sm_rep1_screen.h5" -o 5

   parser.add_option("-d", "--dir", dest="dir", help="test output directory")
    parser.add_option("-s","--savefile",dest = "savefile", help = "Save the 
model to this pickle file", default=None)
    parser.add_option("-r","--restorefile",dest = "restorefile", help = 
"Restore the model from this pickle file", default=None)
    parser.add_option("-i", "--inputfile", dest="inputfile", help="the data 
(hdf5 file) prepended with an absolute path")
    parser.add_option("-c", "--corruption", dest="corruption", type="float", 
help="use this amount of corruption for the dA")
    parser.add_option("-o", "--offset", dest="offset", type="int", help="use 
this offset for reading input from the hdf5 file")

