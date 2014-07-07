#!/bin/bash

# Batch qsub submission script for model search over SdA layer sizes
# with a given output dimension

offset=0

outputLayer=$1
layerType=$2
loss=$3

for i in {800..1100..100}; # first layer
  do

  # three layer models
  first="$i-100-$outputLayer"
  second="$i-200-$outputLayer"
  ((offset++))
  qsub submit_pretrain_gravity_3layers.sh -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset",OUTDIRSUFFIX="$outputLayer",LAYERTYPES="$layerType",LOSS="$loss"
  sleep 5

  first="$i-300-$outputLayer"
  second="$i-400-$outputLayer"
  qsub submit_pretrain_gravity_3layers.sh -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset",OUTDIRSUFFIX="$outputLayer",LAYERTYPES="$layerType",LOSS="$loss"
  sleep 5

  # four layer models
  for j in {500..900..100}; # second layer
    do 
      # Fix the third layer as 100, 200, 300, 400
      # Submit the first job
      first="$i-$j-400-$outputLayer"  
      second="$i-$j-300-$outputLayer"
      ((offset++))
      qsub submit_pretrain_gravity_4layers.sh -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset",OUTDIRSUFFIX="$outputLayer",LAYERTYPES="$layerType",LOSS="$loss"
      sleep 5
      
      # Submit the next job
      first="$i-$j-200-$outputLayer"  
      second="$i-$j-100-$outputLayer"
      ((offset++))
      qsub submit_pretrain_gravity_4layers.sh -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset",OUTDIRSUFFIX="$outputLayer",LAYERTYPES="$layerType",LOSS="$loss"
      
      # Each pair of jobs needs 25 data chunks, and there are 211 in total.  Reset the offset parameter if necessary.
      if (( offset > 185 )); then 
         offset=0
      fi
    
  done
done 
