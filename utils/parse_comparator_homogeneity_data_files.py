""" Process all comparator homogeneity test .npy files, combine into one dataframe """

import sys, re, os
import numpy as np
import pandas as pd

input_dir = '/data/sda_output_data/homogeneity'

# associate a data file name with a given model
filename_to_model = {'ietdata_gmm.npy': 'isomap', 'ketdata_gmm.npy': 'kpca', 'letdata_gmm.npy': 'lle', 'petdata_gmm.npy': 'pca'}

# Store the contents of each file as a DataFrame, add it to the hyperparam_dfs list.
data_files = []
print "...Processing files"

currdir = os.getcwd()
os.chdir(input_dir)

# for each file: 
for infile in ["ietdata_gmm.npy", "ketdata_gmm.npy", "letdata_gmm.npy", "petdata_gmm.npy"]:
    homog_results = np.load(infile)
    # in homog_results: rows are dimension, columns are replicates.  So homog_results[i,j] is the jth replicate for dimension (i+1) * 10
    f_model = filename_to_model[infile]
    for i,row in enumerate(homog_results):
        # build the one line df, store in list
        dimension = str((i+1)*10)
        f_dict = {"Model": [f_model for j in range(0,row.shape[0])], "Dimension": [dimension for j in range(0,row.shape[0])], "Homogeneity": row}
        data_files.append(pd.DataFrame(data=f_dict))
    
print "...Done"
print "...rbinding DataFrames"
master_df = data_files[0]
for i in xrange(1,len(data_files)):
    master_df = master_df.append(data_files[i])
print "...Done"    
os.chdir(input_dir)
master_df.to_csv(path_or_buf="all_comparator_models.csv",index=False)

