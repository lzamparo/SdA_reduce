""" Read the given .h5 files containing SdA reduced data (for a given dimension), and sample a number of labeled points and pack into a data frame.  """

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

input_dir = '/scratch/z/zhaolei/lzamparo/gpu_models/gb_hybrid_cm/3_layers/reduced_data/'
output_dir = '/scratch/z/zhaolei/lzamparo/sm_rep1_data/dfs/'

try:
  limit = int(sys.argv[1]) # define the number of points to try and sample
  cores = int(sys.argv[2]) # use this many cores
  dimension = sys.argv[3]
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

# inputs for each of the files
files_dict = {'10': "reduce_SdA.1000_100_10.2014-07-29.15:59:12.705796.h5 reduce_SdA.1000_300_10.2014-07-29.16:29:01.593674.h5 reduce_SdA.800_200_10.2014-07-29.16:26:52.251526.h5 reduce_SdA.1100_100_10.2014-07-29.16:08:38.229130.h5 reduce_SdA.900_200_10.2014-07-29.16:01:00.278056.h5",
              '20': "reduce_SdA.900_300_20.2014-07-29.18:04:09.355089.h5 reduce_SdA.800_200_20.2014-07-29.17:54:33.993608.h5 reduce_SdA.1100_100_20.2014-07-29.17:43:01.427113.h5 reduce_SdA.1100_300_20.2014-07-29.17:44:54.000147.h5 reduce_SdA.1000_400_20.2014-07-29.17:41:05.616614.h5",
              '30': "reduce_SdA.900_400_30.2014-07-30.12:30:14.356660.h5 reduce_SdA.1000_400_30.2014-07-30.12:14:19.860421.h5 reduce_SdA.900_100_30.2014-07-30.12:24:31.040876.h5 reduce_SdA.800_400_30.2014-07-30.12:28:30.891435.h5 reduce_SdA.1000_100_30.2014-07-30.12:16:04.832854.h5",
              '40': "reduce_SdA.1000_300_40.2014-07-30.14:44:24.686603.h5 reduce_SdA.900_400_40.2014-07-30.14:54:27.386360.h5 reduce_SdA.800_200_40.2014-07-30.14:43:53.788722.h5 reduce_SdA.1100_100_40.2014-07-30.14:39:16.064727.h5 reduce_SdA.1000_400_40.2014-07-30.15:29:20.505129.h5",
              '50': "reduce_SdA.1000_200_50.2014-07-30.16:42:25.771075.h5 reduce_SdA.900_300_50.2014-07-30.16:39:27.750323.h5 reduce_SdA.1100_300_50.2014-07-30.16:34:43.896636.h5 reduce_SdA.800_400_50.2014-07-30.16:16:57.887608.h5 reduce_SdA.1000_100_50.2014-07-30.16:12:45.977627.h5"}


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

label_dict = {0.: "WT", 1.: "Foci", 2.:"Non-Round nuclei"}  # for labeling in the DF
label_opposites = {0.: (1.,2.), 1.: (0.,2.), 2.: (0.,1.)} # for sampling different pts from the given label.
labels = labels[:,0]
data_frames = []

# grab the input files for this job from files dict
for infile in files_dict[dimension].split():
    os.chdir(os.path.join(input_dir,dimension))
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
    outfile = algorithm_name + "_interlabel.csv"
    master_df.to_csv(path_or_buf=outfile,index=False)              
    



