#!/bin/bash

# Batch qsub submission script for model search over SdA layer sizes 

offset=0
for i in {700..1000..100}; # first layer
  do
  for j in {500..900..100}; # second layer
    do 
      # Fix the third layer as 100, 200, 300, 400
      # Submit the first job
      first="$i-$j-400-50"  
      second="$i-$j-300-50"
      ((offset++))
      qsub submit_grid_pretrain_gravity.sh -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset"
      sleep 5
      
      # Submit the next job
      first="$i-$j-200-50"  
      second="$i-$j-100-50"
      ((offset++))
      qsub submit_grid_pretrain_gravity.sh -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset"
      
      # Reset the offset parameter if the grid is too fine.
      if (( offset > 190 )); then 
         offset=0
      fi
    
  done
done 
