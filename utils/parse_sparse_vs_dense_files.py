""" Process the dense vs sparse experiments test output data files.

Fine-tuning file names look like this: 

hybrid_pretraining_adagradmom_relu_relu_relu700_200_10.2014-07-07.00:34:54.554196
hybrid_pretraining_adagradmom_relu_relu_relu_relu800_900_400_10.2014-07-07.00:30:35.985257
hybrid_pretraining_adagrad_gaussian_bernoulli_bernoulli700_200_10.2014-07-04.18:25:42.755640
hybrid_pretraining_adagrad_gaussian_bernoulli_bernoulli_bernoulli800_900_400_10.2014-07-04.18:28:41.891034
hybrid_pretraining_adagrad_relu_relu_relu700_200_10.2014-07-04.19:19:16.170025
hybrid_pretraining_adagrad_relu_relu_relu_relu800_900_400_10.2014-07-04.19:14:50.430017
hybrid_pretraining_cm_gaussian_bernoulli_bernoulli700_200_10.2014-07-02.12:48:17.105814
hybrid_pretraining_cm_gaussian_bernoulli_bernoulli700_200_10.2014-07-04.12:24:29.238507
hybrid_pretraining_cm_gaussian_bernoulli_bernoulli_bernoulli800_900_400_10.2014-07-02.12:51:07.701646
hybrid_pretraining_cm_gaussian_bernoulli_bernoulli_bernoulli800_900_400_10.2014-07-04.12:27:49.490350
hybrid_pretraining_cm_relu_relu_relu700_200_10.2014-07-03.11:59:40.480707
hybrid_pretraining_cm_relu_relu_relu_relu800_900_400_10.2014-07-03.12:01:19.803789

so: hybrid_pretraining_<method>_<units><arch>.<timing>

The average training error for each epoch and batch is reported:
e.g 

Run on 2014-07-07 00:34:54.555885
Pre-training layer 0, epoch 0, cost  50300.4405688
9.99999974738e-06
...
Pickling the model...
Pre-training layer 1, epoch 0, cost  4033.38435487
9.99999974738e-06...
Hybrid pre-training on layers 1 and below, epoch 0, cost 625.042357387
...

Each layer transition is marked by the line: Pickling the model..."""

import sys, re, os
import numpy as np
import pandas as pd
from collections import OrderedDict

from ggplot import *

# Extract the model name from each filename.
def extract_model_and_params(regex,filename):
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

input_dir = '/data/sda_output_data/test_hyperparam_output'

# read a list of all files in the directory that match model output files
currdir = os.getcwd()
os.chdir(input_dir)
model_files = os.listdir(".")

# compile a regex to extract the model from a given filename
model_and_param = re.compile("hybrid_pretraining_([a-z]+)_([a-z_]+)([\d_]+)")
data_regex = re.compile("[a-zA-Z-]+ ([\d])\, epoch ([\d])\, cost ([\d.]+)")

# Store the contents of each file as a DataFrame, add it to the hyperparam_dfs list.
hyperparam_dfs = []
print "...Processing files"

# for each file: 
for f in model_files:
    validation_model = []
    epoch_list = []
    if not f.startswith("Pre-training"):
        continue
    f_model,h_param = extract_model_and_params(model_and_param, f)

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
print ggplot(three_layer, aes(x='epoch', y='score', color='value')) + \
      geom_line() + \
      facet_wrap("param")
