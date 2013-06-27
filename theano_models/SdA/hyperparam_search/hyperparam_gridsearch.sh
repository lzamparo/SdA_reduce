#!/bin/bash

# Batch qsub submission script for hyperparameter values search
# Search over: momentum, weight decay, corruption, (pre-training) learning rate

offset=0
for i in 0 0.3 0.5; # momentum
  do
  for j in 0.7 0.9; # momentum again
    do 
      ((offset++))
      qsub submit_hypersearch_momentum.sh -v FIRSTMODEL="$i",SECONDMODEL="$j",OFFSET="$offset"
      sleep 5
      
      # Reset the offset parameter if the grid is too fine.
      if (( offset > 190 )); then 
         offset=0
      fi
    
  done
done 

for i in 0.00001 0.0001; # weight-decay
  do
  for j in 0.00005 0.0005; # weight-decay again
    do 
      ((offset++))
      qsub submit_hypersearch_wd.sh -v FIRSTMODEL="$i",SECONDMODEL="$j",OFFSET="$offset"
      sleep 5
      
      # Reset the offset parameter if the grid is too fine.
      if (( offset > 190 )); then 
         offset=0
      fi
    
  done
done 

for i in 0 0.1; # corruption
  do
  for j in 0.2 0.3; # corruption again
    do 
      ((offset++))
      qsub submit_hypersearch_corr.sh -v FIRSTMODEL="$i",SECONDMODEL="$j",OFFSET="$offset"
      sleep 5
      
      # Reset the offset parameter if the grid is too fine.
      if (( offset > 190 )); then 
         offset=0
      fi
    
  done
done 

for i in 0.001 0.0005; # learning rate
  do
  for j in 0.0001 0.00005; # learning rate again
    do 
      ((offset++))
      qsub submit_hypersearch_lr.sh -v FIRSTMODEL="$i",SECONDMODEL="$j",OFFSET="$offset"
      sleep 5
      
      # Reset the offset parameter if the grid is too fine.
      if (( offset > 190 )); then 
         offset=0
      fi
    
  done
done 