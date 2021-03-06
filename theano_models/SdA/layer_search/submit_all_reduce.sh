#!/bin/bash

# Batch data reduction job submission script for 3,5 layer models 

# keep track of the directory where the scripts are located.
currdir=`pwd`

# Submit 3 layer jobs

cd "$SCRATCH/gpu_tests/SdA_results/3_layers/finetune_pkl_files"     
for dim in {10..30..10}
do
    cd "$dim"
    for layertype in {"gb","relu"}
    do
        cd "$layertype"
        three_arr=(`ls .`)
        offset=0
        len=${#three_arr[*]}
        for((i=1; i<=$len; i+=2 ))
        do
            let prev=$i-1
            first=${three_arr[$i]}
            second=${three_arr[$prev]}
            pushd "$currdir"
            qsub submit_reduce_gravity_3layers.sh -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset",EXTENSION="$dim/$layertype"
            popd 
            ((offset+=5))
            sleep 5
            
            # Each pair of jobs needs 30 data chunks, and there are 211 in total.  Reset the offset parameter if necessary.
            if (( offset > 181 )); then 
               offset=0
            fi    
        done            
        cd ..
    done
    cd ..
done

# Submit 5 layer jobs
cd "$SCRATCH/gpu_tests/SdA_results/5_layers/finetune_pkl_files" 

for dim in {10..30..10}
do
    cd "$dim"
    for layertype in {"gb","relu"}
    do
        cd "$layertype"
        five_arr=(`ls .`)
        offset=0
        len=${#five_arr[*]}
        for((i=1; i<=$len; i+=2 ))
        do
            let prev=$i-1
            first=${five_arr[$i]}
            second=${five_arr[$prev]}
            pushd "$currdir"
            qsub submit_reduce_gravity_5layers.sh -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset",EXTENSION="$dim/$layertype"
            popd
            ((offset+=5))
            sleep 5
            
            # Each pair of jobs needs 30 data chunks, and there are 211 in total.  Reset the offset parameter if necessary.
            if (( offset > 181 )); then 
               offset=0
            fi    
        done            
        cd ..
    done
    cd ..
done


