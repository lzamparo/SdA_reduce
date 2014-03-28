#!/bin/bash

# Batch qsub submission script for testing pre-training one 3 and one 5 layer models

offset=0

outputLayer=$1
first="700_100_$outputLayer"
second="900_100_$outputLayer"
qsub submit_pretrain_gravity_3layers.sh -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset",OUTDIRSUFFIX="$outputLayer"
((offset+=5))
sleep 5

#first="700_700_400_200_$outputLayer"
#second="800_900_300_100_$outputLayer"
#qsub submit_pretrain_gravity_5layers.sh -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset",OUTDIRSUFFIX="$outputLayer"
#sleep 5
    