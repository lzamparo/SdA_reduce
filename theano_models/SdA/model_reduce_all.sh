#!/bin/bash

# Batch qsub submission script to produce reduced data for each model of SdA

arr=(`ls $SCRATCH/gpu_models/SdA/finetune_pkl_files/10/`)
len=${#arr[*]}

for((i=1; i<=$len; i+=2 ))
do
    let prev=$i-1
    first=${arr[$i]}
    second=${arr[$prev]}
    qsub submit_reduce_gravity.sh -v FIRSTMODEL="$first",SECONDMODEL="$second"
    sleep 5   
done
