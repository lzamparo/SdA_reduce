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

# Extract the model name from each filename.
def extract_model_and_params(regex,filename):
    match = regex.match(filename)
    if match is not None:
        method, layertypes, arch = match.groups()
        layertype = layertypes.split('_')[0]
        return [method,layertype,arch]

# Extract the layer and cost from a line
def parse_line(line,data_regex):
    match = data_regex.match(line)
    if match is not None:
        return match.groups()
    else:
        return (None, None, None)

input_dir = '/data/sda_output_data/init_exps'
currdir = os.getcwd()

# compile a regex to extract the model from a given filename
model_and_param = re.compile("hybrid_pretraining_([a-z]+)_([a-z_]+)([\d_]+)")
data_regex = re.compile("Pre-training layer (\d), epoch (\d), cost  ([\d.]+)")

# Parse each file and add the resultant DataFrame to this list
df_lists = []

for init in ['dense','sparse']:
    print ("...Processing %s files") % init
    os.chdir(os.path.join(input_dir,init))
    model_files = os.listdir(".")   
    
    for f in model_files:
        epoch_list = []
        layer_list = []
        score_list = []
        if not f.startswith("hybrid_pretraining"):
            continue
        method,layertype,arch = extract_model_and_params(model_and_param, f)
    
        infile = open(f, 'r')
        for line in infile:
            if not line.startswith("Pre-training"):
                continue
            (layer, epoch, err) = parse_line(line,data_regex)
            if epoch is not None:
                epoch_list.append(int(epoch))
                score_list.append(float(err))
                layer_list.append(int(layer))
                
        infile.close()
        
        # build the df, store in list
        arch_list = [arch for i in xrange(len(score_list))]
        init_list = [init for i in xrange(len(score_list))]
        model_list = [layertype for i in xrange(len(score_list))]
        f_dict = {"arch": arch_list, "init": init_list, "units": model_list, "layer": layer_list, "score": score_list, "epoch": epoch_list}
        df_lists.append(pd.DataFrame(data=f_dict))
    
print "...Done"
print "...rbinding DataFrames"
master_df = df_lists[0]
for i in xrange(1,len(df_lists)):
    master_df = master_df.append(df_lists[i])
print "...Done"    

os.chdir(input_dir)
master_df.to_csv(path_or_buf="sparse_vs_dense.csv",index=False)
os.chdir(currdir)