""" Process all the model finetuning output files in the given directory
and produce a lineplot of the top 10 models based on reconstruction error.

Fine-tuning file names look like this: finetune_sda_900_500_100_50.2013-06-16.02:25:26.981658

The average training error for each epoch and batch is reported:
e.g epoch 1, minibatch 1/2051, training error 487.399902 

The validation error over all validation batches is reported at the end of each epoch:
e.g epoch 1, minibatch 2051/2051, validation error 266.505805

Each layer transition is marked by the line: Pickling the model..."""

import sys, re, os
import numpy as np
import pandas as pd
from collections import OrderedDict

from ggplot import *

# Extract the model name from each filename.
def extract_model_name(regex,filename):
    match = regex.match(filename)
    if match is not None:
        return match.groups()[0]

# Extract the layer and cost from a line
def parse_line(line,data_regex):
    match = data_regex.match(line)
    if match is not None:
        return match.groups()
    else:
        return (None, None, None, None, None)

input_dir = '/home/lee/projects/sda_output_data/test_finetune_output_mb_and_valid'

# read a list of all files in the directory that match model output files
currdir = os.getcwd()
os.chdir(input_dir)
model_files = os.listdir(".")

# compile a regex to extract the model from a given filename
model_name = re.compile(".*?sda_([\d_]+)\.*")
data_regex = re.compile("epoch ([\d]+)\, minibatch ([\d])+\/([\d]+)\, ([a-z]+) error ([\d.]+)")

# Store the results of the model search in this dictionary
# keys are model name, values are pandas dataframe type objects
training_dfs = OrderedDict()
validation_dfs = OrderedDict()
print "...Processing files"

# for each file: 
for f in model_files:
    # if this file is a pkl or other file, ignore it
    if not f.startswith("finetune_sda"):
        continue

    # read the file, populate this file's entry in the three dicts
    f_model = extract_model_name(model_name, f)
    if f_model is None:
        continue

    if not training_dfs.has_key(f_model):
        training_dfs[f_model]= OrderedDict()
        validation_dfs[f_model] = OrderedDict()

    infile = open(f, 'r')

    for line in infile:
        if not line.startswith("epoch"):
            continue
        (epoch, mb_index, mb_total, phase, err) = parse_line(line,data_regex)
        if epoch is not None:
            if phase == 'validation':
                validation_dfs[f_model][epoch] = [err]
                continue
            if not training_dfs[f_model].has_key(epoch):
                training_dfs[f_model][epoch] = [err]
            else:
                training_dfs[f_model][epoch].append(err)
    infile.close()

print "...Done"