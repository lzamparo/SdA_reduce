""" Process all the SdA homogeneity tests in the given directory
and produce a ranking of top 10 models based on mean homogeneity results.

Files names look like this: 900_500_100_50.npy

Each .npy file contains an nd-array with shape = (1,#iters) """

import sys, re, os
import numpy as np
from collections import OrderedDict

# get the directory from sys.argv[1]
input_dir = sys.argv[1]

# read a list of all files in the directory that match model output files
os.chdir(input_dir)
model_files = os.listdir(".")

# Store the results of the model search in this dictionary
# keys are model name, values are mean homogeneity scores
results = {}

print "...Processing files"

# for each file: 
for f in model_files:
    # if this file is not an .npy file, ignore it
    if not f.endswith(".npy"):
        continue
    
    # read the file, populate results dict with mean homogeneity value
    parts = f.split('.')
    f_model = parts[0]
    if f_model is None:
        continue
    
    homog_results = np.load(f)
    results[f_model] = homog_results.mean()       
    
print "...Done"
    
# At this point, find the top 10 scoring models in each of the dicts
# Also, compute some order statistics to qualify this list: max, min
print "...Finding top 10 models by mean homogeneity score"

print "Top ten archs: " 

sorted_layer_results = sorted(results.items(), key=lambda t: t[1], reverse=True)
for i in range(0,20):
    model, score = sorted_layer_results[i]
    print str(i) + ": " + model + " , " + str(score)
    
sl_max = max(results.values())
sl_min = min(results.values())
print "Max, min, mean " + ": " + str(sl_max) + " , " + str(sl_min) + " , " + str(np.mean(results.values()))        