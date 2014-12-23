""" Process all the model finetuning output files in the given directory
and produce a ranking of top 20 models based on reconstruction error.

Files names look like this: finetune_sda_900_500_100_50.2013-06-16.02:25:26.981658

The average training error for each layer of each model is reported:

e.g epoch 198, minibatch 59/59, validation error 424.194899 %

Each layer transition is marked by the line: Pickling the model..."""

import sys, re, os
import numpy as np
from collections import OrderedDict

# Extract the model name from each filename.
def extract_model_name(regex,filename):
    match = regex.match(filename)
    if match is not None:
        return match.groups()[0]

# Extract the layer and cost from a line
def extract_cost(regex,line):
    parts = line.split(" ")
    try:
        match = regex.match(line)
        if match is not None:
            return match.groups()[0]
        else:
            return 9999.
    except:
        print "Value error in trying to match: " + line

# get the directory from sys.argv[1]
input_dir = sys.argv[1]

# read a list of all files in the directory that match model output files
os.chdir(input_dir)
model_files = os.listdir(".")

# compile a regex to extract the model from a given filename
model_name = re.compile(".*?sda_([\d_]+)\.*")
get_error = re.compile("validation error ([\d\.]+)")

# Store the results of the model search in this dictionary
# keys are model name, values are validation scores
results = {}

print "...Processing files"

# for each file: 
for f in model_files:
    # if this file is a pkl or other file, ignore it
    if not f.startswith("finetune_sda"):
        continue
    
    # read the file, populate this file's entry in the three dicts
    f_model = extract_model_name(model_name, f)
    results[f_model] = 9999.
    if f_model is None:
        continue
    
    infile = open(f, 'r')
    for line in infile:
        if not line.startswith("epoch"):
            continue
        cost = extract_cost(get_error,line)
        if float(cost) < results[f_model]:
            results[f_model] = float(cost)        
    infile.close()
    
print "...Done"
    
# At this point, find the top 10 scoring models in each of the dicts
# Also, compute some order statistics to qualify this list: max, min
print "...Finding top 10 scoring results"

print "Top ten archs: " 

sorted_layer_results = sorted(results.items(), key=lambda t: t[1])
for i in range(0,10):
    model, score = sorted_layer_results[i]
    print str(i) + ": " + model + " , " + str(score)
    
sl_max = max(results.values())
sl_min = min(results.values())
print "Max, min, mean " + ": " + str(sl_max) + " , " + str(sl_min) + " , " + str(np.mean(results.values()))        