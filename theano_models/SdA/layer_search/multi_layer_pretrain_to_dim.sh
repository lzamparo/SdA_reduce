#!/bin/bash

# Batch qsub submission script for pre-training 3,5 layer models with architectures specified in the following lists 
# with a given final layer size parameter, layer types

offset=0

threeLayerModelList=(700_100_ 700_200_ 600_300_ 900_400_
700_400_ 500_400_ 500_300_ 700_300_ 700_250_
700_350_ 700_450_ 500_300_ 900_300_ 900_350_
900_200_ 600_400_ 700_100_ 400_300_ 900_100_
650_400_)

fiveLayerModelList=(700_800_700_100_ 1000_850_700_200_ 1000_800_600_300_ 800_850_900_400_
1000_850_700_400_ 700_600_500_400_ 700_600_500_300_ 700_750_700_300_ 900_800_700_200_
700_800_500_300_ 700_700_400_200_ 900_600_500_300_ 800_900_500_300_ 1000_900_500_300_
900_800_400_200_ 700_600_400_200_ 900_700_300_100_ 900_700_400_300_ 700_900_400_100_
900_700_400_200_)

outputLayer=$1

layerType=$2

loss=$3

len=${#threeLayerModelList[*]}

for((i=1; i<=$len; i+=2 ))
do
    let prev=$i-1
    first="${threeLayerModelList[$i]}$outputLayer"
    second="${threeLayerModelList[$prev]}$outputLayer"
    qsub submit_pretrain_gravity_3layers.sh -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset",OUTDIRSUFFIX="$outputLayer",LAYERTYPES="$layerType",LOSS="$loss"
    ((offset+=5))
    sleep 5
    
    # Each pair of jobs needs 30 data chunks, and there are 211 in total.  Reset the offset parameter if necessary.
    if (( offset > 181 )); then 
       offset=0
    fi    
done

len=${#fiveLayerModelList[*]}

for((i=1; i<=$len; i+=2 ))
do
    let prev=$i-1
    first="${fiveLayerModelList[$i]}$outputLayer"
    second="${fiveLayerModelList[$prev]}$outputLayer"
    qsub submit_pretrain_gravity_5layers.sh -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset",OUTDIRSUFFIX="$outputLayer",LAYERTYPES="$layerType",LOSS="$loss" 
    ((offset+=5))
    sleep 5
    
    # Each pair of jobs needs 30 data chunks, and there are 211 in total.  Reset the offset parameter if necessary.
    if (( offset > 181 )); then 
       offset=0
    fi    
done
