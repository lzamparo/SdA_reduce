#!/bin/bash

# Batch job submission script for GaussianBernoulli dA versus ReLU dA experiments

# submit the dA experiments
qsub submit_dA_foci.sh -v CORRUPTION=0.00
sleep 5
qsub submit_dA_foci.sh -v CORRUPTION=0.05
sleep 5
qsub submit_dA_foci.sh -v CORRUPTION=0.10
sleep 5
qsub submit_dA_foci.sh -v CORRUPTION=0.15
sleep 5  
qsub submit_dA_foci.sh -v CORRUPTION=0.20
sleep 5  
qsub submit_dA_foci.sh -v CORRUPTION=0.25
sleep 5  
qsub submit_dA_foci.sh -v CORRUPTION=0.30
sleep 5  
qsub submit_dA_foci.sh -v CORRUPTION=0.35
sleep 5
qsub submit_dA_foci.sh -v CORRUPTION=0.40
sleep 5

