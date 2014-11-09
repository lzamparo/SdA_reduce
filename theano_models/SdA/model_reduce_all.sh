#!/bin/bash

# Batch qsub submission script to produce reduced data for each model of SdA

dim=$1

#three_arr=(`ls $SCRATCH/gpu_models/gb_hybrid_cm/3_layers/finetune_pkl_files/$dim/`)
#len=${#three_arr[*]}
#dir="${SCRATCH}/gpu_models/gb_hybrid_cm/3_layers/reduced_data"

#for((i=1; i<=$len; i+=2 ))
#do
    #let prev=$i-1
    #first=${three_arr[$i]}
    #second=${three_arr[$prev]}
    #qsub -v FIRSTMODEL="$first",SECONDMODEL="$second",DIR="$dir",DIM="$dim" submit_reduce_gravity.sh
    #sleep 5   
#done

four_arr=(`ls $SCRATCH/gpu_models/gb_hybrid_cm/4_layers/finetune_pkl_files/$dim/`)
len=${#four_arr[*]}
dir="${SCRATCH}/gpu_models/gb_hybrid_cm/4_layers/reduced_data"

for((i=1; i<=$len; i+=2 ))
do
    let prev=$i-1
    first=${four_arr[$i]}
    second=${four_arr[$prev]}
    qsub -v FIRSTMODEL="$first",SECONDMODEL="$second",DIR="$dir",DIM="$dim" submit_reduce_gravity.sh
    sleep 5   
done
