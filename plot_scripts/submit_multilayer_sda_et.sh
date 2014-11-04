#!/bin/bash

# Batch SdA embedding test job submission script for 3,4 layer models 

# keep track of the directory where the scripts are located.
currdir=`pwd`

# Submit 3 layer jobs

cd "$SCRATCH/gpu_models/gb_hybrid_cm/3_layers/reduced_data"     
for dim in {10..50..10}
do
    cd "$dim"
    do
        three_arr=(`ls *.h5`)
        basedir=`pwd`
        for model in ${three_arr[@]}
        do
            # send reduced base dir, reduced file, output dir with job
            pushd "$currdir"
            qsub submit_sda_et.sh -v BASEDIR=$basedir,OUTPUTDIR=$basedir,REDUCEDFILE=$model
            popd 
            sleep 5    
        done            
    done
    cd ..
done

# Submit 5 layer jobs
cd "$SCRATCH/gpu_models/gb_hybrid_cm/5_layers/reduced_data" 

for dim in {10..50..10}
do
    cd "$dim"
    do
        five_arr=(`ls *.h5`)
        basedir=`pwd`
        for model in ${five_arr[@]}
        do
            # send reduced base dir, reduced file, output dir with job
            pushd "$currdir"
            qsub submit_sda_et.sh -v BASEDIR=$basedir,OUTPUTDIR=$basedir,REDUCEDFILE=$model
            popd 
            sleep 5  
        done            
    done
    cd ..
done


