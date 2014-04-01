#!/bin/bash

# Batch qsub submission script for testing finetuning one pair of 3 layer and one pair of 5 layer models

offset=0

outputLayer=$1
first="900_300_$outputLayer"
second="500_300_$outputLayer"
qsub submit_finetune_gravity_3layers.sh -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset"
((offset+=5))
sleep 5

first="700_700_400_200_$outputLayer"
second="800_900_300_100_$outputLayer"
qsub submit_finetune_gravity_5layers.sh -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset"
sleep 5
    
