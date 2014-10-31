# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from common_density_plot_utils import *

#################### The script part to generate the plots, and find the limits ####################

import pickle as pkl
from numpy.random import shuffle

# Grab the headers for the AreaShape features
as_header_file = open("/data/NOP10/Cells_headers.txt")
as_list = as_header_file.readlines()
as_header_file.close()
as_headers = [item.strip().split()[1] for item in as_list]
as_positions = [item.strip().split()[0] for item in as_list]

# Select data from samples using the as_headers as well as 'Label'
as_headers.append('Label')

# Grab the labeled data, randomly sub-sample one of each labeled files with stratification  
labeled_examples_pd = pd.DataFrame.from_csv('/data/NOP10/Phenotypes_Nucleolus_Samples_TS2.csv',index_col=False)
label_groups = labeled_examples_pd.groupby('Label')['FileName']
indices = [shuffle(v) for k, v in label_groups.groups.iteritems()]
indices = [v[0] for k, v in label_groups.groups.iteritems()]


sample_labeled_files = labeled_examples_pd.iloc[indices,:]
labeled_files = pd.unique(sample_labeled_files['FileName'])
plates = pd.unique(sample_labeled_files['Plate'])

# Grab the data for what labeled FileNames we have, keep only those 
data_reader = pd.read_csv('/data/NOP10/SQL_Image_Object_GeneNames_Merged_TS2_NoGhosts.csv',index_col=5,iterator=True,chunksize=50000)
labeled_data = None
for chunk in data_reader:
    chunk['ImageNumber'] = chunk.index
    #labeled_file_pts = chunk[chunk['FileName'].isin(labeled_files) & chunk['Plate'].isin()]
    labeled_file_pts = pd.merge(chunk, sample_labeled_files, on=["Plate","FileName"])
    
    # skip chunks with no data from the files we've selected
    if len(labeled_file_pts) == 0:
        continue
    
    # merge the labeled_file_pts with the labels of their matching FileNames
    #labeled_data_pts = pd.merge(labeled_file_pts, labeled_files, on='FileName')
    
    if labeled_data is None:
        labeled_data = labeled_file_pts.loc[:,as_headers]
    else:
        labeled_data = labeled_data.append(labeled_file_pts.loc[:,as_headers], ignore_index=True)
    

# Go through the features, calculate the thresholds
thresholds = {}
as_headers.remove("Label")
for feature in as_headers:
    wt_mean = labeled_data[feature].where(labeled_data['Label'] == 'negative').mean()
    wt_std = labeled_data[feature].where(labeled_data['Label'] == 'negative').std()
    lower,upper = wt_mean - 2*wt_std, wt_mean + 2*wt_std
    thresholds[feature] = (lower,upper)

# Pickle the thresholds, along with their column positions
filename = as_headers[0].split('_')[0] + "_" + "nop10"+ "_" + "thresholds.pkl"
pkl.dump((zip(as_positions,as_headers),thresholds), open(filename,'wb'))

# Pickle the labeled_data sample
filename = "NOP10_labeled_df.pkl"
pkl.dump((labeled_data),open(filename,'wb'))

####################  Plot the data and thresholds ####################
(ph, thresholds) = pkl.load(open("Cells_nop10_thresholds.pkl", mode='rb'))
labeled_data = pkl.load(open("NOP10_labeled_df.pkl", mode='rb'))

# We only care about these labels
labels_used = np.unique(labeled_data['Label']).tolist()

# Try a faceted density plot for each feature
fig = plt.figure(figsize=(24,11))
for n,key in enumerate(thresholds.keys()):
    lower,upper = thresholds[key]
    sp = fig.add_subplot(2,7,n+1)
    x_vals = make_x_axis(labeled_data[labeled_data['Label'] == "negative"][key])
    # plot all labels worth of densities, as well as the thresholds
    for label in labels_used:
        data = labeled_data[labeled_data['Label'] == label][key]
        kde = make_kde(data)
        rfill_between(sp, x_vals, kde(x_vals),label)
    sp.set_title(key.split('_')[-1])
    sp.axvline(lower,ls='--',color='k')
    sp.axvline(upper,ls='--',color='k')
    rstyle(sp)

# Put a legend below current axis
sp.legend(loc='upper center', bbox_to_anchor=(-3.35, -0.05),
          fancybox=True, shadow=True, ncol=len(labels_used)/2)

# Put a title on the main figure
fig.suptitle("NOP10: Area and Shape Parameter Density Plots by Label (with 2 x std WT dashed)",fontsize=20)
fig.subplots_adjust(left=.03, right=.97, top=0.91,hspace=0.14,wspace=0.27)
plt.show()     
