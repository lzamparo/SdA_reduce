""" Read all the .h5 files and .npy files in the directories below input, and compute the mean and var of class-wise distance distributions """

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

input_dir = '/scratch/z/zhaolei/lzamparo/sm_rep1_data'
output_dir = '/scratch/z/zhaolei/lzamparo/sm_rep1_data/dfs'

try:
  limit = int(sys.argv[1]) # define the number of points to try and sample
  cores = int(sys.argv[2]) # use this many cores
  infile = sys.argv[3] # use this .h5 file as input
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


print("...Processing top level .h5 files")

os.chdir(input_dir)

label_dict = {0.: "WT", 1.: "Foci", 2.:"Non-Round nuclei"}
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
    for i in xrange(1,len(label_dfs)):
        master_df = master_df.append(label_dfs[i])
    print("...Done")    
    os.chdir(output_dir)
    outfile = algorithm_name + "_" + dims + ".csv"
    master_df.to_csv(path_or_buf=outfile,index=False)       
data_set_file.close()            
    
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
    
#print("...Done") 
#print("...rbinding DataFrames")

#master_df = data_files[0]
#for i in xrange(1,len(data_files)):
    #master_df = master_df.append(data_files[i])
    
#print("...Done")    
#os.chdir(output_dir)
#master_df.to_csv(path_or_buf="all_sda_models.csv",index=False)

