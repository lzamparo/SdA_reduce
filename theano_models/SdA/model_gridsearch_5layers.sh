#!/bin/bash

# Batch qsub submission script for model search over 5 layer SdA models
# with a given output dimension

offset=0

outputLayer=$1
layerType=$2
loss=$3

for i in {1000..1500..100}; # first layer
do

  for j in {500..900..100}; # second layer
  do

	for k in {100..400..100}; # third layer
	do

	  for l in {40..90..10}; # fourth layer
	  do
      	  first="$i-$j-$k-$l-$outputLayer"
      	  second="$i-$j-$k-$l-$outputLayer"
      	  ((offset++))
      	  qsub -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset",OUTDIRSUFFIX="$outputLayer",LAYERTYPES="$layerType",LOSS="$loss" submit_pretrain_gravity_5layers.sh
      	  sleep 5

          # Each pair of jobs needs 30 data chunks, and there are 211 in total.  Reset the offset parameter if necessary.
          if (( offset > 180 )); then 
             offset=0
          fi
      done
    done
  done
done
