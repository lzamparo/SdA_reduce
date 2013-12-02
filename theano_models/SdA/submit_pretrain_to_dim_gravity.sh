#!/bin/bash

# Batch qsub submission script for pre-training a set of models specified in a list 
# with a given final layer size parameter.

offset=0

modelList=(700_700_100_ 1000_700_200_ 1000_600_300_ 800_900_400_
1000_700_400_ 700_500_400_ 700_500_300_ 700_700_300_ 900_700_200_
700_800_300_ 700_700_200_ 900_500_300_ 800_900_300_ 1000_900_300_
900_800_200_ 700_600_400_ 900_700_100_ 900_700_300_ 700_900_100_
900_700_400_)

outputLayer=$1

len=${#modelList[*]}

for((i=1; i<=$len; i+=2 ))
do
    let prev=$i-1
    first="${modelList[$i]}$outputLayer"
    second="${modelList[$prev]}$outputLayer"
    echo submit_pretrain_gravity.sh -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset",OUTDIRSUFFIX="$outputLayer"
    ((offset+=5))
    sleep 5
    
    # Each pair of jobs needs 30 data chunks, and there are 211 in total.  Reset the offset parameter if necessary.
    if (( offset > 181 )); then 
       offset=0
    fi    
done
