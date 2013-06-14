#!/bin/bash

# Batch qsub submission script for model search over SdA layer sizes 

arr=(`ls $SCRATCH/gpu_models/SdA/pretrain_pkl_files/`)
offset=0
len=${#arr[*]}

for((i=1; i<=$len; i+=2 ))
do
    let prev=$i-1
    first=${arr[$i]}
    second=${arr[$prev]}
    echo submit_finetune_gravity.sh -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset"
    ((offset+=5))
    sleep 5
    
    # Reset the offset parameter if the grid is too fine.
    if (( offset > 190 )); then 
       offset=0
    fi    
done
