#!/bin/bash

# Batch qsub submission script for testing gradient update values for one pair of 3 layer and one pair of 5 layer models

offset=0

outputLayer=$1
first="SdA_900_300_$outputLayer.pkl"
second="SdA_500_300_$outputLayer.pkl"
qsub submit_test_gradient_gravity_3layers.sh -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset"
((offset+=5))
sleep 5

first="SdA_700_700_400_200_$outputLayer.pkl"
second="SdA_800_900_500_300_$outputLayer.pkl"
qsub submit_test_gradient_gravity_5layers.sh -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset"
sleep 5
    
