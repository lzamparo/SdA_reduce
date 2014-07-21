#!/bin/bash

# Batch qsub submission script for finetuning hyperparameter values search
# Search over: momentum, weight decay, learning rate

offset=0
for i in 0.1 0.3 0.5 0.7 0.9; # momentum
do 
  ((offset++))
  qsub submit_hypersearch_momentum.sh -v MOMENTUM="$i",OFFSET="$offset"
  sleep 5
  
  # Reset the offset parameter if the grid is too fine.
  if (( offset > 190 )); then 
     offset=0
  fi
done 

for i in 0.00001 0.00005 0.0001 0.0005 0.001 0.005; # weight-decay
do
  ((offset++))
  qsub submit_hypersearch_wd.sh -v WEIGHTDECAY="$i",OFFSET="$offset"
  sleep 5
  
  # Reset the offset parameter if the grid is too fine.
  if (( offset > 190 )); then 
     offset=0
  fi
done 

for i in 0.01 0.005 0.001 0.0005 0.0001 0.00005; # learning rate
do
  ((offset++))
  qsub submit_hypersearch_lr.sh -v LEARNINGRATE="$i",OFFSET="$offset"
  sleep 5
  
  # Reset the offset parameter if the grid is too fine.
  if (( offset > 190 )); then 
     offset=0
  fi
done 