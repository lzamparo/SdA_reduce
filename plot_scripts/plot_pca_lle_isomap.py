"""
==========
PCA vs LLE vs ISOMAP
==========

This script performs PCA, LLE, and ISOMAP and plots the data projected down onto two dimensions, in at attempt to 
evaluate the differences between the three algorithms on a selection from an image screen data set.  

Adapted from the example provided by Mathieu Blondel: http://scikit-learn.org/stable/_downloads/plot_kernel_pca.py

"""

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D

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

pca = PCA()
D_pca = pca.fit_transform(D_scaled)

# Plot results.  Manipulate D_scaled by standardizing the PCA matrix to shrink
# plotting axis if required.

n_samples, n_features = D.shape
n_neighbors = 30

#----------------------------------------------------------------------
# Isomap projection 
print "Computing Isomap embedding"
t0 = time()
D_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(D_scaled)
print "Done in time %.2fs " % (time() - t0)

#----------------------------------------------------------------------
# Locally linear embedding 
print "Computing LLE embedding"
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='standard')
t0 = time()
D_lle = clf.fit_transform(D_scaled)
print "Done in time %.2fs " % (time() - t0)
print "Reconstruction error: %g" % clf.reconstruction_error_

if opts.dimension == 2:
    pl.figure(figsize=(12,8),dpi=100)
    plot_embedding(D_pca, 1, "PCA projection")
    plot_embedding(D_iso, 2, "Isomap projection")
    plot_embedding(D_lle, 3, "LLE projection")
    pl.subplots_adjust(left=None, bottom=None, right=None, wspace=0.15, hspace=None)
    pl.savefig(opts.outputfile,format="eps", orientation='landscape', pad_inches=0)    
else:
    # Twice as wide as it is tall.
    fig = plt.figure(figsize=plt.figaspect(0.5))    
    plot_embedding_3D(D_iso, 1, "Isomap projection")
    plot_embedding_3D(D_lle, 2, "Local Linear Embedding")
    plt.savefig("manifold_fig_3D.pdf",format="pdf",dpi=200, orientation='landscape', pad_inches=0)
    plt.show()


#----------------------------------------------------------------------
# Scale and visualize the embedding vectors in 2D
def plot_embedding(X, tile, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    pl.subplot(1, 3, tile)
    
    # Establish the indices for plotting as slices of the X matrix
    # Only need the foci upper index, all others can be sliced using the dimensions already stored
    foci_upper_index = wt_samplesize + foci_samplesize
    
    pl.plot(X[:wt_samplesize, 0], X[:wt_samplesize, 1], "ro")
    pl.plot(X[wt_samplesize:foci_upper_index, 0], X[wt_samplesize:foci_upper_index, 1], "bo")
    pl.plot(X[foci_upper_index:, 0], X[foci_upper_index:, 1], "go")
          
    legend_font_props = FontProperties()
    legend_font_props.set_size('small')
    pl.legend( ('Wild Type', 'Foci', 'Non-round Nuclei'), loc="lower left", numpoints=1, prop=legend_font_props)
    
    #pl.xticks([]), pl.yticks([])
    if title is not None:
        pl.title(title,fontsize=15)

# close the data file
datafile.close()
