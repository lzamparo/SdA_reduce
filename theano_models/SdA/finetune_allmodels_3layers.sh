#!/bin/bash

# Batch qsub submission script for multi-layer models

outputLayer=$1

# Submit 3 layer jobs

three_arr=(`ls $SCRATCH/gpu_models/more_features/3_layers/pretrain_pkl_files/$outputLayer/`)
offset=0
len=${#three_arr[*]}

for((i=1; i<=$len; i+=2 ))
do
    let prev=$i-1
    first=${three_arr[$i]}
    second=${three_arr[$prev]}
    qsub -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset" submit_finetune_gravity_3layers.sh
    ((offset+=5))
    sleep 5

    # Each pair of jobs needs 30 data chunks, and there are 211 in total.  Reset the offset parameter if necessary.
    if (( offset > 181 )); then
       offset=0
    fi
done

