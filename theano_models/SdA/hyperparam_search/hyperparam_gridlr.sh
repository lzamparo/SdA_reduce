#!/bin/bash

# Batch qsub submission script for learning rate hyperparameter values search

offset=0

for i in 0.001 0.0005; # learning rate
  do
  for j in 0.0001 0.0005; # learning rate again
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
