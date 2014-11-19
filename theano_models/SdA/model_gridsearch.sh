#!/bin/bash

# Batch qsub submission script for model search over SdA layer sizes
# with a given output dimension

offset=0

outputLayer=$1
layerType=$2
loss=$3

for i in {1000..1500..100}; # first layer
  do

  # three layer models
  first="$i-100-$outputLayer"
  second="$i-200-$outputLayer"
  ((offset++))
  qsub -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset",OUTDIRSUFFIX="$outputLayer",LAYERTYPES="$layerType",LOSS="$loss"  submit_pretrain_gravity_3layers.sh
  sleep 5

  first="$i-300-$outputLayer"
  second="$i-400-$outputLayer"
  qsub -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset",OUTDIRSUFFIX="$outputLayer",LAYERTYPES="$layerType",LOSS="$loss" submit_pretrain_gravity_3layers.sh
  sleep 5

  # four layer models
  for j in {500..900..100}; # second layer
    do 
      # Fix the third layer as 100, 200, 300, 400
      # Submit the first job
      first="$i-$j-400-$outputLayer"  
      second="$i-$j-300-$outputLayer"
      ((offset++))
      qsub -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset",OUTDIRSUFFIX="$outputLayer",LAYERTYPES="$layerType",LOSS="$loss" submit_pretrain_gravity_4layers.sh
      sleep 5
      
      # Submit the next job
      first="$i-$j-200-$outputLayer"  
      second="$i-$j-100-$outputLayer"
      ((offset++))
      qsub -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset",OUTDIRSUFFIX="$outputLayer",LAYERTYPES="$layerType",LOSS="$loss" submit_pretrain_gravity_4layers.sh
      
      # Each pair of jobs needs 30 data chunks, and there are 211 in total.  Reset the offset parameter if necessary.
      if (( offset > 180 )); then 
         offset=0
      fi
    
  done
done 
