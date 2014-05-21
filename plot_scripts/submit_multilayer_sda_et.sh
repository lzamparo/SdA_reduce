#!/bin/bash

# Batch SdA embedding test job submission script for 3,5 layer models 

# keep track of the directory where the scripts are located.
currdir=`pwd`

# Submit 3 layer jobs

cd "$SCRATCH/gpu_tests/SdA_results/3_layers/reduced_data"     
for dim in {10..30..10}
do
    cd "$dim"
    for layertype in {"gb","relu"}
    do
        cd "$layertype"
        three_arr=(`ls *.h5`)
        basedir=`pwd`
        for model in ${three_arr[@]}
        do
            # send reduced base dir, reduced file, output dir with job
            pushd "$currdir"
            echo "qsub submit_sda_et.sh -v BASEDIR=$basedir,OUTPUTDIR=$basedir,REDUCEDFILE=$model" 
            popd 
            sleep 5    
        done            
        cd ..
    done
    cd ..
done

# Submit 5 layer jobs
cd "$SCRATCH/gpu_tests/SdA_results/5_layers/reduced_data" 

for dim in {10..30..10}
do
    cd "$dim"
    for layertype in {"gb","relu"}
    do
        cd "$layertype"
        five_arr=(`ls *.h5`)
        basedir=`pwd`
        for model in ${five_arr[@]}
        do
            # send reduced base dir, reduced file, output dir with job
            pushd "$currdir"
            echo "qsub submit_sda_et.sh -v BASEDIR=$basedir,OUTPUTDIR=$basedir,REDUCEDFILE=$model" 
            popd 
            sleep 5  
        done            
        cd ..
    done
    cd ..
done


