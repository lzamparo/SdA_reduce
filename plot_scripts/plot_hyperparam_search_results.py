""" Process all the model finetuning hyperparameter test output files.

Fine-tuning file names look like this: hyperparam_search<model architecture>_<hyperparam>.2014-07-22.00:02:37.242738

The average training error for each epoch and batch is reported:
e.g epoch 1, minibatch 1/2051, training error 487.399902 

The validation error over all validation batches is reported at the end of each epoch:
e.g epoch 1, minibatch 2051/2051, validation error 266.505805

For hyperparameters, we care only about the validation error over the epochs.

Each layer transition is marked by the line: Pickling the model..."""

import sys, re, os
import numpy as np
import pandas as pd
from collections import OrderedDict

from ggplot import *

# Extract the model name from each filename.
def extract_model_and_param(regex,filename):
    match = regex.match(filename)
    if match is not None:
        return match.groups()

# Extract the layer and cost from a line
def parse_line(line,data_regex):
    match = data_regex.match(line)
    if match is not None:
        return match.groups()
    else:
        return (None, None, None, None, None)

input_dir = '/home/lee/projects/sda_output_data/test_hyperparam_output'

# read a list of all files in the directory that match model output files
currdir = os.getcwd()
os.chdir(input_dir)
model_files = os.listdir(".")

# compile a regex to extract the model from a given filename
model_and_param = re.compile("hyperparam_search([\d_]+)_([\w]+)\.*")
data_regex = re.compile("epoch ([\d]+)\, minibatch ([\d])+\/([\d]+)\, ([a-z]+) error ([\d.]+)")

# Store the contents of each file as a DataFrame, add it to the hyperparam_dfs list.
hyperparam_dfs = []
print "...Processing files"

# for each file: 
for f in model_files:
    validation_model = []
    epoch_list = []
    if not f.startswith("hyperparam_search"):
        continue
    f_model,h_param = extract_model_and_param(model_and_param, f)

    infile = open(f, 'r')
    for line in infile:
        if line.startswith(h_param):
            h_value = float(line.strip().split()[1])
        if not line.startswith("epoch"):
            continue
        (epoch, mb_index, mb_total, phase, err) = parse_line(line,data_regex)
        if epoch is not None and phase == 'validation':
            epoch_list.append(int(epoch))
            validation_model.append(float(err))
            
    infile.close()
    
    # build the df, store in list
    model_list = [f_model for i in xrange(len(validation_model))]
    h_param_list = [h_param for i in xrange(len(validation_model))]
    h_value_list = [str(h_value) for i in xrange(len(validation_model))]
    f_dict = {"model": model_list, "param": h_param_list, "value": h_value_list, "score": validation_model, "epoch": epoch_list}
    hyperparam_dfs.append(pd.DataFrame(data=f_dict))
    
print "...Done"
print "...rbinding DataFrames"
master_df = hyperparam_dfs[0]
for i in xrange(1,len(hyperparam_dfs)):
    master_df = master_df.append(hyperparam_dfs[i])
print "...Done"    

three_layer = master_df[master_df.model == '1000_400_20']
three_layer = three_layer[["epoch","score","value","param"]]
four_layer = master_df[master_df.model != '1000_400_20']
four_layer = four_layer[["epoch","score","value","param"]]

three_layer.to_csv(path_or_buf="three_layer_model.csv")
four_layer.to_csv(path_or_buf="four_layer_model.csv")
master_df.to_csv(path_or_buf="both_models.csv")
#print ggplot(three_layer, aes(x='epoch', y='score', color='value')) + \
#      geom_line() + \
#      facet_wrap("param")
