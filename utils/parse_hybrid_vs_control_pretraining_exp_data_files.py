""" Process all the hybrid pretraining vs standard pretraining output files.

the directory structure looks like:
base -
    hybrid -
        3_layers -
            finetune_output -
                10 -
                    gb -
                    relu -
                20 -
                    gb -
                    relu -
        4_layers -
            finetune_output -
                10 -
                    gb -
                    relu -
                20 -
                    gb -
                    relu -
    control -
        3_layers -
            finetune_output -
                10 -
                    gb -
                    relu -
                20 -
                    gb -
                    relu -
        4_layers -
            finetune_output -
                10 -
                    gb -
                    relu -
                20 -
                    gb -
                    relu -
        

Fine-tuning file names look like this: finetune_sda_<model_arch>.2014-06-18.23:25:20.199112

The average training error for each epoch and batch is reported:
e.g epoch 50, minibatch 2620/2620, training error 124.029259

The validation error over all validation batches is reported at the end of each epoch:
e.g epoch 50, minibatch 2620/2620, validation error 169.730011

We care only about the validation error over the epochs.

Each layer transition is marked by the line: Pickling the model..."""

import sys, re, os
import numpy as np
import pandas as pd
from collections import OrderedDict

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

input_dir = '/data/sda_output_data/pretraining_hybrid_vs_control'

# compile a regex to extract the model from a given filename
model_and_param = re.compile("finetune_sda_([\d_]+).[\w]+\.*")
data_regex = re.compile("epoch ([\d]+)\, minibatch ([\d])+\/([\d]+)\, ([a-z]+) error ([\d.]+)")

# Store the contents of each file as a DataFrame, add it to the hyperparam_dfs list.
data_files = []
print "...Processing files"

currdir = os.getcwd()

# for each file: 
for method in ["hybrid","control"]:
    for group in ["3_layers","4_layers"]:
        for layertype in ["gb","relu"]:
            for dimension in ['10','20']:
                # read a list of all files in the directory that match model output files
                os.chdir(os.path.join(input_dir,method,group,'finetune_output',dimension,layertype))
                model_files = os.listdir(".")        
                for f in model_files:
                    validation_model = []
                    epoch_list = []
                    if not f.startswith("finetune_sda"):
                        continue
                    f_model = extract_model_and_param(model_and_param, f)
                
                    infile = open(f, 'r')
                    for line in infile:
                        if not line.startswith("epoch"):
                            continue
                        (epoch, mb_index, mb_total, phase, err) = parse_line(line,data_regex)
                        if epoch is not None and phase == 'validation':
                            epoch_list.append(int(epoch))
                            validation_model.append(float(err))
                            
                    infile.close()
                    
                    # build the df, store in list
                    method_list = [method for i in xrange(len(validation_model))]
                    model_list = [f_model[0] for i in xrange(len(validation_model))]
                    group_list = [group for i in xrange(len(validation_model))]
                    layer_list = [layertype for i in xrange(len(validation_model))]
                    dimension_list = [dimension for i in xrange(len(validation_model))]
                    f_dict = {"method": method_list,"model": model_list, "group": group_list, "layer": layer_list, "dimension": dimension_list,"score": validation_model, "epoch": epoch_list}
                    data_files.append(pd.DataFrame(data=f_dict))
    
print "...Done"
print "...rbinding DataFrames"
master_df = data_files[0]
for i in xrange(1,len(data_files)):
    master_df = master_df.append(data_files[i])
print "...Done"    
os.chdir(input_dir)
master_df.to_csv(path_or_buf="both_methods.csv",index=False)

