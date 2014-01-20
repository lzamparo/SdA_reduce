"""
==========
PCA vs LLE vs ISOMAP
==========

This script performs PCA, LLE, and ISOMAP and plots the data projected down onto two dimensions, in at attempt to 
evaluate the differences between the three algorithms on a selection from an image screen data set.  

Adapted from the example provided by Mathieu Blondel: http://scikit-learn.org/stable/_downloads/plot_kernel_pca.py

"""

import numpy as np
import matplotlib as mpl
mpl.use('pdf')			# needed so that you can plot in a batch job with no X server (undefined $DISPLAY) problems 

import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.font_manager import FontProperties
#from mpl_toolkits.mplot3d import Axes3D

import logging
import sys
from tables import *
from optparse import OptionParser
from time import time

from extract_datasets import extract_labeled_chunkrange
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.preprocessing import scale

np.random.seed(0)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--h5file",
              dest="inputfile", help="Read data input from this hdf5 file.")
op.add_option("--size",
              dest="size", type="int", help="Extract the first size chunks of the data set and labels.")
op.add_option("--sample-size",
              dest="samplesize", type="int", help="The max size of the samples")
op.add_option("--dimension",
              dest="dimension", type="int", help="Produce a plot in this number of dimensions (either 2 or 3)")
op.add_option("--output",
              dest="outputfile", help="Write the plot to this output file.")

(opts, args) = op.parse_args()

fig = plt.figure(figsize=(12,6),dpi=100)

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors in 2D
def plot_embedding(X, tile, rescale=None, legend_loc="lower left", title=None):
    
    if rescale is not None:
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

    sub = fig.add_subplot(1, 3, tile)
    
    # Establish the indices for plotting as slices of the X matrix
    # Only need the foci upper index, all others can be sliced using the dimensions already stored
    foci_upper_index = wt_samplesize + foci_samplesize
    
    sub.plot(X[:wt_samplesize, 0], X[:wt_samplesize, 1], "ro")
    sub.plot(X[wt_samplesize:foci_upper_index, 0], X[wt_samplesize:foci_upper_index, 1], "bo")
    sub.plot(X[foci_upper_index:, 0], X[foci_upper_index:, 1], "go")
          
    legend_font_props = FontProperties()
    legend_font_props.set_size('small')
    sub.legend( ('Wild Type', 'Foci', 'Non-round Nuclei'), loc=legend_loc, numpoints=1,prop=legend_font_props)
    
    if title is not None:
        sub.set_title(title,fontsize=15)


###############################################################################
# Load a training set from the given .h5 file
datafile = openFile(opts.inputfile, mode = "r", title = "Data is stored here")

# Extract some of the dataset from the datafile
X, labels = extract_labeled_chunkrange(datafile, opts.size)

# Sample from the dataset
wt_labels = np.nonzero(labels[:,0] == 0)[0]
foci_labels = np.nonzero(labels[:,0] == 1)[0]
ab_nuclei_labels = np.nonzero(labels[:,0] == 2)[0]

wt_data = X[wt_labels,5:]
foci_data = X[foci_labels,5:]
ab_nuclei_data = X[ab_nuclei_labels,5:]

# Figure out the sample sizes based on the shape of the *_labels arrays and the 
# sample size argument

wt_samplesize = min(opts.samplesize,wt_data.shape[0])
foci_samplesize = min(opts.samplesize,foci_data.shape[0])
ab_nuclei_samplesize = min(opts.samplesize, ab_nuclei_data.shape[0]) 

# can just use np.random.permutation(array)[0:size,:] to sample u at random
# from the strata.
wt_data_sample = np.random.permutation(wt_data)[0:wt_samplesize,:]
foci_data_sample = np.random.permutation(foci_data)[0:foci_samplesize,:]
ab_nuclei_sample = np.random.permutation(ab_nuclei_data)[0:ab_nuclei_samplesize,:]

D = np.vstack((wt_data_sample,foci_data_sample,ab_nuclei_sample))
D_scaled = scale(D)

#----------------------------------------------------------------------
# PCA projection
print "Computing PCA embedding"
pca = PCA()
D_pca = pca.fit_transform(D_scaled)

n_samples, n_features = D.shape
n_neighbors = 30

#----------------------------------------------------------------------
# Isomap projection 
print "Computing Isomap embedding"
t0 = time()
D_iso = Isomap(n_neighbors, out_dim=2).fit_transform(D_scaled)
print "Done in time %.2fs " % (time() - t0)

#----------------------------------------------------------------------
# Locally linear embedding 
print "Computing LLE embedding"
clf = LocallyLinearEmbedding(n_neighbors, out_dim=2,
                                      method='modified')
t0 = time()
D_lle = clf.fit_transform(D_scaled)
print "Done in time %.2fs " % (time() - t0)
print "Reconstruction error: %g" % clf.reconstruction_error_

plot_embedding(D_pca, 1, rescale="yes", title="PCA projection", legend_loc="lower left")
plot_embedding(D_iso, 2, rescale="yes",title="Isomap projection", legend_loc="lower left")
plot_embedding(D_lle, 3, rescale=None, title="LLE projection", legend_loc="upper left")
fig.savefig(opts.outputfile,format="pdf", orientation='landscape', pad_inches=0)    

# close the data file
datafile.close()



