#!/bin/bash

# Batch qsub submission script for 3,5 layer models 

# Submit 3 layer jobs

three_arr=(`ls $SCRATCH/gpu_tests/SdA_results/3_layers/pretrain_pkl_files/10/gb`)
offset=0
len=${#three_arr[*]}

for((i=1; i<=$len; i+=2 ))
do
    let prev=$i-1
    first=${three_arr[$i]}
    second=${three_arr[$prev]}
    qsub submit_finetune_gravity_3layers.sh -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset"
    ((offset+=5))
    sleep 5
    
    # Each pair of jobs needs 30 data chunks, and there are 211 in total.  Reset the offset parameter if necessary.
    if (( offset > 181 )); then 
       offset=0
    fi    
done

# Submit 5 layer jobs

five_arr=(`ls $SCRATCH/gpu_tests/SdA_results/5_layers/pretrain_pkl_files/10/gb`)
offset=0
len=${#five_arr[*]}

for((i=1; i<=$len; i+=2 ))
do
    let prev=$i-1
    first=${five_arr[$i]}
    second=${five_arr[$prev]}
    qsub submit_finetune_gravity_5layers.sh -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset"
    ((offset+=5))
    sleep 5
    
    # Each pair of jobs needs 30 data chunks, and there are 211 in total.  Reset the offset parameter if necessary.
    if (( offset > 181 )); then 
       offset=0
    fi    
done


