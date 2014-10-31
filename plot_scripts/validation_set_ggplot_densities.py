# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from common_density_plot_utils import *


#################### The script part to generate the plots, and find the limits ####################

from tables import *
from extract_datasets import extract_labeled_chunkrange    
import pickle as pkl

# AreaShape feature names for both Cells and Nuclei; choose one for reduction
# I removed EulerNumber,Orientation from either Nuclei or Cells thresholds; they're uninformative
cells_areashape_names = ["Cells_AreaShape_Area","Cells_AreaShape_Eccentricity","Cells_AreaShape_Solidity","Cells_AreaShape_Extent","Cells_AreaShape_Perimeter","Cells_AreaShape_FormFactor","Cells_AreaShape_MajorAxisLength","Cells_AreaShape_MinorAxisLength"]
nuclei_areashape_names = ["Nuclei_AreaShape_Area","Nuclei_AreaShape_Eccentricity","Nuclei_AreaShape_Solidity","Nuclei_AreaShape_Extent","Nuclei_AreaShape_Perimeter","Nuclei_AreaShape_FormFactor","Nuclei_AreaShape_MajorAxisLength","Nuclei_AreaShape_MinorAxisLength"]

# Grab the headers for the AreaShape features
header_file = open('/data/sm_rep1_screen/Object_Headers_trimmed.txt')
headers = header_file.readlines()
headers = [item.strip() for item in headers]
positions = [headers.index(name) for name in cells_areashape_names]
labeled_shape_data_headers = [headers[pos] for pos in positions]
header_file.close()

# Grab the validation data and labels, select only those positions we want
data_file = openFile('/data/sm_rep1_screen/sample.h5','r')
nodes = data_file.listNodes('/recarrays')
data,labels = extract_labeled_chunkrange(data_file,11)
labels = labels[:,0]
label_names = {0.: 'WT', 1.: "Focus", 2.: "Non-round nucleus", 3.: "Bizarro"}
label_str = [label_names[val] for val in labels]
shape_data = data[:,positions]
data_file.close()

# Form & concatenate the label DF with the data DF
labels_pd = pd.DataFrame({'labels': label_str})
data = pd.DataFrame(shape_data, columns=labeled_shape_data_headers)
labeled_data = pd.concat([labels_pd,data],axis=1)

# Go through the features, calculate the thresholds
thresholds = {}
for feature in labeled_shape_data_headers:
    wt_mean = labeled_data[feature].where(labeled_data['labels'] == 'WT').mean()
    wt_std = labeled_data[feature].where(labeled_data['labels'] == 'WT').std()
    lower,upper = wt_mean - 2*wt_std, wt_mean + 2*wt_std
    thresholds[feature] = (lower,upper)

# Pickle the thresholds, along with their column positions
filename = labeled_shape_data_headers[0].split('_')[0] + "_" + "thresholds.pkl"
pkl.dump((zip(positions,labeled_shape_data_headers),thresholds), open(filename,'wb'))


####################  Plot the data and thresholds ####################

# We only care about these labels
labels_used = ["WT", "Focus", "Non-round nucleus"]

# Try a faceted density plot for each feature
fig = plt.figure()
for n,key in enumerate(thresholds.keys()):
    lower,upper = thresholds[key]
    sp = fig.add_subplot(4,2,n+1)
    x_vals = make_x_axis(labeled_data[labeled_data['labels'] == "WT"][key])
    # plot all labels worth of densities, as well as the thresholds
    for label in labels_used:
        data = labeled_data[labeled_data['labels'] == label][key]
        kde = make_kde(data)
        rfill_between(sp, x_vals, kde(x_vals),label)
    sp.set_title(key)
    sp.axvline(lower,ls='--',color='k')
    sp.axvline(upper,ls='--',color='k')
    rstyle(sp)

# Put a legend below current axis
sp.legend(loc='upper center', bbox_to_anchor=(-0.25, -0.15),
          fancybox=True, shadow=True, ncol=4)

# Put a title on the main figure
fig.suptitle("Area and Shape Parameter Density Plots by Label (with 2 x std WT dashed)",fontsize=20)
fig.subplots_adjust(hspace=0.35)
plt.show()     
