""" Read all the .h5 files and .npy files in the directories below input, and compute the mean and var of class-wise distance distributions """

import sys, re, os
import numpy as np
import pandas as pd

from collections import OrderedDict
from tables import *
from extract_datasets import extract_unlabeled_byarray
from scipy.spatial.distance import pdist

import contextlib,time
@contextlib.contextmanager
def timeit():
  t=time.time()
  yield
  print(time.time()-t,"sec")

input_dir = '/data/sda_output_data/homogeneity/csv_data'
output_dir = '/data/sda_output_data/homogeneity/csv_data/dfs'

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


print "...Processing top level .h5 files"

os.chdir(input_dir)

label_dict = {0.: "WT", 1.: "Foci", 2.:"Non-Round nuclei"}
labels = labels[:,0]

data_frames = []
count = 0  # index for the line-by-line df.  I feel dirty.
for infile in ["isomap_data.h5", "kpca_data.h5", "lle_data.h5", "pca_data.h5"]:
    print "...processing " + infile
    data_set_file = openFile(infile,'r')
    for dims in ["dim10","dim20","dim30","dim40","dim50"]:
        data = data_set_file.getNode("/recarrays", dims)
        cutoff = min([data.shape[0],labels.shape[0]])
        labeled_data = np.hstack((labels[:cutoff,np.newaxis],data[:cutoff,:]))
        
        # split on label, select elements, calculate distance matrix, shove mean & var into DF
        used_labels = np.unique(labels)[0:3]
        for label in used_labels:
            subset = labeled_data[labeled_data[:,0] == label,1:]   # select all those elements with this label
            with timeit():
              distances = pdist(subset)
            data_frames.append(pd.DataFrame({"mean": distances.mean(), "var": distances.var(), "dimension": dims, "label": label_dict[label], "algorithm": infile.split("_")[0]}, index=[count]))
            count = count + 1

    data_set_file.close()             

master_df = data_frames[0]
for i in xrange(1,len(data_frames)):
    master_df = master_df.append(data_frames[i])
print "...Done"    
os.chdir(output_dir)
master_df.to_csv(path_or_buf="comparators_euclidean.csv",index=False)        
        
    
    
#data_frames = []
#print "...Processing 3,4 layer SdA .npy files"

## for each SdA .npy file:
#for group in ["3_layers","4_layers"]:
    #for dimension in ['10','20','30','40','50']:
        ## read a list of all files in the directory that match model output files
        #os.chdir(os.path.join(input_dir,group,dimension))
        #model_files = os.listdir(".")        
        #for f in model_files:
            #if not f.endswith(".npy"):
                #continue
            #data = np.load(infile)
            #model_name = infile.split("gmm")[0]
            #nrows = data.shape[0]
            ## build the df, store in list
            #model_list = [model_name for i in xrange(nrows)]
            #group_list = [group for i in xrange(nrows)]
            #dimension_list = [dimension for i in xrange(nrows)]
            #f_dict = {"model": model_list, "group": group_list, "dimension": dimension_list}
            #data_files.append(pd.concat([pd.DataFrame(data=f_dict),pd.DataFrame(data)],axis=1))
    
#print "...Done"
#print "...rbinding DataFrames"

#master_df = data_files[0]
#for i in xrange(1,len(data_files)):
    #master_df = master_df.append(data_files[i])
    
#print "...Done"    
#os.chdir(output_dir)
#master_df.to_csv(path_or_buf="all_sda_models.csv",index=False)

