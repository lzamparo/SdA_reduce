#! /bin/bash

first="700_200_10"
second="800_900_400_10"
let offset=0
layerType="relu"
let sparsity=15
dirsuffix="sparse"
loss="squared"

qsub submit_test_hybrid_multiproc.sh -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset",LAYERTYPES="$layerType",LOSS="$loss",SPARSITY="$sparsity",DIRSUFFIX="$dirsuffix"

sleep 5

let sparsity=-1
dirsuffix="dense"

qsub submit_test_hybrid_multiproc.sh -v FIRSTMODEL="$first",SECONDMODEL="$second",OFFSET="$offset",LAYERTYPES="$layerType",LOSS="$loss",SPARSITY="$sparsity",DIRSUFFIX="$dirsuffix"
