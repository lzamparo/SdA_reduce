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
    qsub submit_finetune_gravity.sh -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset"
    ((offset+=5))
    sleep 5
    
    # Each pair of jobs needs 30 data chunks, and there are 211 in total.  Reset the offset parameter if necessary.
    if (( offset > 181 )); then 
       offset=0
    fi    
done
