""" Read all the .h5 files and .npy files in the directories below input, and compute the list of class-wise distance distributions """

import sys, re, os
import numpy as np
import pandas as pd
from collections import OrderedDict
from tables import *
from extract_datasets import extract_unlabeled_byarray

input_dir = '/data/sda_output_data/homogeneity'
output_dir = '/data/sda_output_data/homogeneity/csv_data'

print "...Grabbing the labels for the validation set"
start = 0
data_set_file = openFile('/data/sm_rep1_screen/sample.h5')
labels_list = data_set_file.listNodes("/labels", classname='Array')
labels = np.empty(labels_list[start].shape)
empty = True
for labelnode in labels_list:
    if empty:
        labels[:] = labelnode.read()
        empty = False
    else:
        labels = np.vstack((labels,labelnode.read()))
data_set_file.close()
labels = labels[:,0]

print "...Processing top level .h5 files"
os.chdir(input_dir)

for infile in ["isomap_data.h5", "kpca_data.h5", "lle_data.h5", "pca_data.h5"]:
    data_set_file = openFile(infile,'r')
    for dims in [1,2,3,4,5]:
        data = extract_unlabeled_byarray(data_set_file, dims)
        cutoff = min([data.shape[0],labels.shape[0]])
        labeled_data = np.hstack((labels[:cutoff],data[:cutoff,:]))
        # TODO: make into pandas DataFrame, split on label, calculate distance matrix, take lower triangle, shove into list
        # will have three lists per algorithm * dimension.  I can store this in a tidy data style: ravel each lower triangle,
        # form into DataFrame with colnames = {algorithm, dimension, label, distance}
        
    data_set_file.close()
    
data_file = []
print "...Processing 3,4 layer SdA .npy files"

# for each SdA .npy file:
for group in ["3_layers","4_layers"]:
    for dimension in ['10','20','30','40','50']:
        # read a list of all files in the directory that match model output files
        os.chdir(os.path.join(input_dir,group,dimension))
        model_files = os.listdir(".")        
        for f in model_files:
            if not f.endswith(".npy"):
                continue
            data = np.load(infile)
            model_name = infile.split("gmm")[0]
            nrows = data.shape[0]
            # build the df, store in list
            model_list = [model_name for i in xrange(nrows)]
            group_list = [group for i in xrange(nrows)]
            dimension_list = [dimension for i in xrange(nrows)]
            f_dict = {"model": model_list, "group": group_list, "dimension": dimension_list}
            data_files.append(pd.concat([pd.DataFrame(data=f_dict),pd.DataFrame(data)],axis=1))
    
print "...Done"
print "...rbinding DataFrames"

master_df = data_files[0]
for i in xrange(1,len(data_files)):
    master_df = master_df.append(data_files[i])
    
print "...Done"    
os.chdir(output_dir)
master_df.to_csv(path_or_buf="all_sda_models.csv",index=False)

