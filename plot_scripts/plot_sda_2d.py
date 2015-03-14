"""
==========
SdA 2d visualizations
==========

This script selects a stratified sample from a validation set and plots it.  The input data resides in .h5 files from a given directory
After first grabbing one file and calculating the indices of points to sample, go through and plot those sampled points for each h5 file.

"""

import numpy as np
import matplotlib as mpl
mpl.use('pdf')			# needed so that you can plot in a batch job with no X server (undefined $DISPLAY) problems 

import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.font_manager import FontProperties

import logging
import sys, os, re
import tables
from optparse import OptionParser
from time import time

sys.path.append('/home/lee/projects/SdA_reduce/utils')
from extract_datasets import extract_labeled_chunkrange

np.random.seed(0)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--h5dir",
              dest="directory", help="Read data input files from this directory.")
op.add_option("--size",
              dest="size", type="int", help="Extract the first size chunks of the data set and labels.")
op.add_option("--sample-size",
              dest="samplesize", type="int", help="The max size of the samples")
op.add_option("--output",
              dest="outputfile", help="Write the plot to this output file.")

(opts, args) = op.parse_args()

def extract_name(filename,regex):
    model_name = regex.match(filename)
    return model_name.groups()[0]

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors in 2D
def plot_embedding(X, tile, sizes, title=None):
    #x_min, x_max = np.min(X, 0), np.max(X, 0)
    #X = (X - x_min) / (x_max - x_min)

    sub = fig.add_subplot(2, 5, tile)

    # Establish the indices for plotting as slices of the X matrix
    # Only need the foci upper index, all others can be sliced using the dimensions already stored
    wt_samplesize, foci_samplesize, ab_nuclei_samplesize = sizes
    foci_upper_index = wt_samplesize + foci_samplesize
    
    sub.plot(X[:wt_samplesize, 0], X[:wt_samplesize, 1], "ro")
    sub.plot(X[wt_samplesize:foci_upper_index, 0], X[wt_samplesize:foci_upper_index, 1], "bo")
    sub.plot(X[foci_upper_index:, 0], X[foci_upper_index:, 1], "go")
          
    #legend_font_props = FontProperties()
    #legend_font_props.set_size('small')
    #sub.legend( ('Wild Type', 'Foci', 'Non-round Nuclei'), loc="lower left", numpoints=1,prop=legend_font_props)
    
    if title is not None:
        sub.set_title(title,fontsize=17)
        
    return sub


# This becomes a subroutine to extract the data from each h5 file in the directory.  
def sample_from(file_name, opts, sampling_tuple=None):
    ''' Return the sample from the data, the size of each sample, and a tuple containing the rows sampled for each label. '''
    
    datafile = tables.openFile(file_name, mode = "r", title = "Data is stored here")
    
    # Extract some of the dataset from the datafile
    X, labels = extract_labeled_chunkrange(datafile, opts.size)
    
    # Sample from the dataset
    wt_labels = np.nonzero(labels[:,0] == 0)[0]
    foci_labels = np.nonzero(labels[:,0] == 1)[0]
    ab_nuclei_labels = np.nonzero(labels[:,0] == 2)[0]
    
    wt_data = X[wt_labels,:]
    foci_data = X[foci_labels,:]
    ab_nuclei_data = X[ab_nuclei_labels,:]
    
    # Figure out the sample sizes based on the shape of the *_labels arrays and the 
    # sample size argument
    
    wt_samplesize = min(opts.samplesize,wt_data.shape[0])
    foci_samplesize = min(opts.samplesize,foci_data.shape[0])
    ab_nuclei_samplesize = min(opts.samplesize, ab_nuclei_data.shape[0]) 
    sizes = (wt_samplesize, foci_samplesize, ab_nuclei_samplesize)
    
    if sampling_tuple is None:
        # stratified sampling from each 
        wt_rows = np.arange(wt_data.shape[0])
        foci_rows = np.arange(foci_data.shape[0])
        ab_nuclei_rows = np.arange(ab_nuclei_data.shape[0])
        
        np.random.shuffle(wt_rows)
        np.random.shuffle(foci_rows)
        np.random.shuffle(ab_nuclei_rows)
        
        sampling_tuple = (wt_rows,foci_rows,ab_nuclei_rows)
    
    else:
        wt_rows,foci_rows,ab_nuclei_rows = sampling_tuple
        
    wt_data_sample = wt_data[wt_rows[:wt_samplesize],:]
    foci_data_sample = foci_data[foci_rows[:foci_samplesize],:]
    ab_nuclei_sample = ab_nuclei_data[ab_nuclei_rows[:ab_nuclei_samplesize],:]
    
    D = np.vstack((wt_data_sample,foci_data_sample,ab_nuclei_sample))
    datafile.close()
    return D, sampling_tuple, sizes

# Read all h5 files
os.chdir(opts.directory)
files = [f for f in os.listdir('.') if f.endswith('.h5')]
name_pattern = re.compile('reduce_SdA\.([\d_]+)\.[\d]+')
sampling_tuple = None

fig = plt.figure(figsize=(20,10),dpi=100)
for tile,f in enumerate(files):
    data, sampling_tuple, sizes = sample_from(f, opts, sampling_tuple)
    sub = plot_embedding(data, tile, sizes, extract_name(f, name_pattern))

# Put a legend below current axis
legend_font_props = FontProperties()
legend_font_props.set_size('large')
sub.legend( ('Wild Type', 'Foci', 'Non-round Nuclei'), loc="lower left", numpoints=1,prop=legend_font_props, bbox_to_anchor=(-1.85, -0.20),
              fancybox=True, shadow=True, ncol=3)

# Put a title on the main figure
#fig.suptitle("2D projections of 10 different SdA models",fontsize=20)
fig.subplots_adjust(hspace=0.25)

# Save the figure
fig.savefig(opts.outputfile,format="pdf", orientation='landscape', pad_inches=0)    



