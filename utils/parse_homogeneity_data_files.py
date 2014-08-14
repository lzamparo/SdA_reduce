""" Process all the SdA homogeneity test .npy files, combine into a dataframe """

import sys, re, os
import numpy as np
import pandas as pd

# Extract the model name from each filename.
def extract_model(regex,filename):
    match = regex.match(filename)
    if match is not None:
        return match.groups()[0]

input_dir = '/data/sda_output_data/homogeneity'

# compile a regex to extract the model from a given filename
model_and_param = re.compile("^([\d_]+)")

# Store the contents of each file as a DataFrame, add it to the hyperparam_dfs list.
data_files = []
print "...Processing files"

currdir = os.getcwd()
# for each file: 
for layers in ["3_layers","4_layers"]:
    for dimension in ["10","20","30","40","50"]:
        # read a list of all files in the directory that match model output files
        os.chdir(os.path.join(input_dir,layers,dimension))
        model_files = os.listdir(".")  
        for f in model_files:
            if not f.endswith(".npy"):
                continue
            f_model = extract_model(model_and_param, f)
            infile = open(f, 'r')
            homog_results = np.load(f)                 
            infile.close()
    
            # build the one line df, store in list
            f_dict = {"Model": [f_model], "Layers": [layers], "Dimension": [dimension], "Homogeneity": [homog_results.mean()]}
            data_files.append(pd.DataFrame(data=f_dict))
    
print "...Done"
print "...rbinding DataFrames"
master_df = data_files[0]
for i in xrange(1,len(data_files)):
    master_df = master_df.append(data_files[i])
print "...Done"    
os.chdir(input_dir)
master_df.to_csv(path_or_buf="all_sda_models.csv",index=False)

