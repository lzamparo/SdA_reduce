""" Read all the .h5 files in the directories below input, sample from classes and compute inter-class distances """

### N.B: this is meant to run in python 3!

import sys, re, os
import numpy as np
import pandas as pd

from collections import OrderedDict
from tables import *
from sklearn.metrics.pairwise import euclidean_distances

import contextlib,time
@contextlib.contextmanager
def timeit():
  t=time.time()
  yield
  print(time.time()-t,"sec")

input_dir = '/scratch/z/zhaolei/lzamparo/sm_rep1_data'
output_dir = '/scratch/z/zhaolei/lzamparo/sm_rep1_data/dfs'

try:
  limit = int(sys.argv[1]) # define the number of points to try and sample
  cores = int(sys.argv[2]) # use this many cores
  infile = sys.argv[3] # use this .h5 file as input
except IndexError:
  limit = 500 # the default.  The lazy man's arg_parse().
  cores = 8

def make_sample_df(labels, np, labeled_data, limit, algorithm_name, dims, cores):
  used_labels = np.unique(labels)[0:3]
  label_dfs = []
  label = used_labels[0]
  
  # sub-sample the stratified subset
  subset = labeled_data[labeled_data[:,0] == label,1:]   # select all those elements with this label
  num_samples = min(limit,subset.shape[0])
  indices = np.arange(subset.shape[0])
  np.random.shuffle(indices)
  label_pts = subset[indices[:num_samples],:]
  
  # repeat for the same number of pts from one opposing label
  first_comparators = labeled_data[labeled_data[:,0] == label_opposites[label][0],1:]
  num_samples = min(limit,first_comparators.shape[0])
  indices = np.arange(first_comparators.shape[0])
  np.random.shuffle(indices)
  opposing_pts = first_comparators[indices[:num_samples],:]      
  distances = euclidean_distances(label_pts,opposing_pts)
  num_records = distances.size      
  label_dfs.append(pd.DataFrame({"distances": distances.ravel(), "dimension": [dims for i in range(num_records)], "label": [label_dict[label] for i in range(num_records)], "opposing label": [label_dict[label_opposites[label][0]] for i in range(num_records)], "algorithm": [algorithm_name for i in range(num_records)]}))      
  
  # repeat for the same number of pts from the other opposing label
  second_comparators = labeled_data[labeled_data[:,0] == label_opposites[label][1],1:]
  num_samples = min(limit,second_comparators.shape[0])
  indices = np.arange(second_comparators.shape[0])
  np.random.shuffle(indices)
  opposing_pts = second_comparators[indices[:num_samples],:]      
  distances = euclidean_distances(label_pts,opposing_pts)
  num_records = distances.size      
  label_dfs.append(pd.DataFrame({"distances": distances.ravel(), "dimension": [dims for i in range(num_records)], "label": [label_dict[label] for i in range(num_records)], "opposing label": [label_dict[label_opposites[label][1]] for i in range(num_records)], "algorithm": [algorithm_name for i in range(num_records)]}))       
      
  return label_dfs


print("...Grabbing the labels for the validation set")
start = 0
data_set_file = openFile('/scratch/z/zhaolei/lzamparo/sm_rep1_data/sample.h5')
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


print("...Processing top level .h5 files")

os.chdir(input_dir)

label_dict = {0.: "WT", 1.: "Foci", 2.:"Non-Round nuclei"} # for labeling in the DF
label_opposites = {0.: (1.,2.), 1.: (0.,2.), 2.: (0.,1.)} # for sampling different pts from the given label.
labels = labels[:,0]
data_frames = []

print("...processing ", infile)
algorithm_name = infile.split("_")[0]
os.chdir(input_dir)
data_set_file = openFile(infile,'r')
for dims in ["dim10","dim20","dim30","dim40","dim50"]:
    data = data_set_file.getNode("/recarrays", dims)
    cutoff = min([data.shape[0],labels.shape[0]])
    labeled_data = np.hstack((labels[:cutoff,np.newaxis],data[:cutoff,:]))
    
    # split on label, select elements, calculate distance matrix, shove mean & var into DF
    with timeit():
      label_dfs = make_sample_df(labels, np, labeled_data, limit, algorithm_name, dims, cores)
    
    # write to file
    master_df = label_dfs[0]
    for i in range(1,len(label_dfs)):
        master_df = master_df.append(label_dfs[i])
    print("...Done")    
    os.chdir(output_dir)
    outfile = algorithm_name + "_" + dims + "_interlabel.csv"
    master_df.to_csv(path_or_buf=outfile,index=False)       
data_set_file.close()            
    



