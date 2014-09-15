""" Read the given .h5 files containing SdA reduced data (for a given dimension), and sample a number of labeled points and pack into a data frame.  """

### N.B: this is meant to run in python 3!

import sys, re, os
import numpy as np
import pandas as pd

from collections import OrderedDict
from tables import *
from sklearn.metrics.pairwise import pairwise_distances

import contextlib,time
@contextlib.contextmanager
def timeit():
  t=time.time()
  yield
  print(time.time()-t,"sec")

input_dir = '/scratch/z/zhaolei/lzamparo/gpu_models/gb_hybrid_cm/3_layers/reduced_data/'
output_dir = '/scratch/z/zhaolei/lzamparo/sm_rep1_data/dfs/'

try:
  limit = int(sys.argv[1]) # define the number of points to try and sample
  cores = int(sys.argv[2]) # use this many cores
  dimension = sys.argv[3]
  infiles = sys.argv[4:] # use this .h5 file as input
except IndexError:
  limit = 1000 # the default.  The lazy man's arg_parse().
  cores = 8

def make_sample_df(labels, np, labeled_data, limit, algorithm_name, dims, cores):
  used_labels = np.unique(labels)[0:3]
  label_dfs = []
  for label in used_labels:
    
      subset = labeled_data[labeled_data[:,0] == label,1:]   # select all those elements with this label
      # sub-sample the stratified subset
      num_samples = min(limit,subset.shape[0])
      indices = np.arange(subset.shape[0])
      np.random.shuffle(indices)
      sampled_pts = subset[indices[:num_samples],:]        
      distances = pairwise_distances(sampled_pts, n_jobs=-1)
      lt_distances = np.tril(distances,k=-1)
      distances_nozeros = lt_distances[lt_distances != 0.].ravel()
      num_records = distances_nozeros.shape[0]
      label_dfs.append(pd.DataFrame({"distances": distances_nozeros, "dimension": [dims for i in range(num_records)], "label": [label_dict[label] for i in range(num_records)], "algorithm": [algorithm_name for i in range(num_records)]}))
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

print("...Processing SdA .h5 files")

os.chdir(input_dir)

label_dict = {0.: "WT", 1.: "Foci", 2.:"Non-Round nuclei"}
labels = labels[:,0]
data_frames = []

for infile in infiles:
    os.chdir(input_dir)
    print("...processing ", infile)
    algorithm_name = infile.split(".")[1] # file names look like reduce_SdA.1000_100_10.2014-07-29.15:59:12.705796.h5, we want the model ID

    data_set_file = openFile(infile,'r')
    
    # grab all the data
    empty = True
    nodes_list = data_set_file.listNodes("/recarrays")
    data = np.empty(nodes_list[0].shape)
    for node in nodes_list:
      if empty:
        data[:] = node.read()
        empty = False
      else:
        data = np.vstack((data,node.read()))
    data_set_file.close()   
    
    cutoff = min([data.shape[0],labels.shape[0]])
    labeled_data = np.hstack((labels[:cutoff,np.newaxis],data[:cutoff,:]))
    
    # split on label, select elements, calculate distance matrix, shove mean & var into DF
    with timeit():
      label_dfs = make_sample_df(labels, np, labeled_data, limit, algorithm_name, dimension, cores)
    
    # write to file
    master_df = label_dfs[0]
    for i in range(1,len(label_dfs)):
        master_df = master_df.append(label_dfs[i])
    print("...Done")    
    os.chdir(output_dir)
    outfile = algorithm_name + "_" + dimension + ".csv"
    master_df.to_csv(path_or_buf=outfile,index=False)              
    



