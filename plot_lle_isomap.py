"""
=============================================================================
Manifold learning on cell measurements: Locally Linear Embedding, Isomap.
=============================================================================

Perform stratified sampling on data derived from measurements of cell images.
Algorithms used: LLE and Isomap.
"""

# Adapted from the digits example provided by Fabian Pedregosa <fabian.pedregosa@inria.fr>, Olivier Grisel <olivier.grisel@ensta.org>, Mathieu Blondel <mathieu@mblondel.org> 
# see: http://scikit-learn.org/stable/_downloads/plot_lle_digits.py                 
# License: BSD

print __doc__
from time import time

# matplotlib imports
import pylab as pl
from matplotlib import offsetbox
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties


# Numpy and sklearn imports
from sklearn import manifold, decomposition
from sklearn import preprocessing
import numpy as np

# Data handling imports
from tables import *
from optparse import OptionParser
import utils.extract_datasets

# parse commandline arguments
op = OptionParser()
op.add_option("--h5file",
              dest="inputfile", help="Read data input from this hdf5 file.")
op.add_option("--size",
              dest="size", type="int", help="Extract the first size chunks of the data set and labels.")
op.add_option("--sample-size",
              dest="samplesize", type="int", help="The max size of the samples")
op.add_option("--dimension", 
              dest="dimension", type="int", help="Specifies the dimension of the lower dimensional space")
(opts, args) = op.parse_args()

# 
###############################################################################
# Load a training set from the given .h5 file
datafile = openFile(opts.inputfile, mode = "r", title = "Data is stored here")

# Extract some of the dataset from the datafile
X, labels = utils.extract_datasets.extract_datasets(datafile, opts.size)

# Sample from the dataset
wt_labels = np.nonzero(labels[:,0] == 0)[0]
foci_labels = np.nonzero(labels[:,0] == 1)[0]
ab_nuclei_labels = np.nonzero(labels[:,0] == 2)[0]

# Discard the first 4 columns of bookeeping info
wt_data = X[wt_labels,5:]
foci_data = X[foci_labels,5:]
ab_nuclei_data = X[ab_nuclei_labels,5:]

# Figure out the sample sizes based on the shape of the *_labels arrays and the 
# sample size argument
wt_samplesize = min(opts.samplesize,wt_data.shape[0])
foci_samplesize = min(opts.samplesize,foci_data.shape[0])
ab_nuclei_samplesize = min(opts.samplesize, ab_nuclei_data.shape[0]) 

wt_data_sample = np.random.permutation(wt_data)[0:wt_samplesize,:]
foci_data_sample = np.random.permutation(foci_data)[0:foci_samplesize,:]
ab_nuclei_sample = np.random.permutation(ab_nuclei_data)[0:ab_nuclei_samplesize,:]

# Put the sample data together and re-scale the data.
D = np.vstack((wt_data_sample,foci_data_sample,ab_nuclei_sample))
D_scaled = preprocessing.scale(D)

n_samples, n_features = D.shape
n_neighbors = 30


#----------------------------------------------------------------------
# Scale and visualize the embedding vectors in 2D
def plot_embedding(X, tile, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    pl.subplot(1, 2, tile)
    
    # Establish the indices for plotting as slices of the X matrix
    # Only need the foci upper index, all others can be sliced using the dimensions already stored
    foci_upper_index = wt_samplesize + foci_samplesize
    
    pl.plot(X[:wt_samplesize, 0], X[:wt_samplesize, 1], "ro")
    pl.plot(X[wt_samplesize:foci_upper_index, 0], X[wt_samplesize:foci_upper_index, 1], "bo")
    pl.plot(X[foci_upper_index:, 0], X[foci_upper_index:, 1], "go")
          
    legend_font_props = FontProperties()
    legend_font_props.set_size('small')
    pl.legend( ('Wild Type', 'Foci', 'Non-round Nuclei'), loc="lower left", numpoints=1, prop=legend_font_props)
    
    pl.xticks([]), pl.yticks([])
    if title is not None:
        pl.title(title,fontsize=15)
        
#----------------------------------------------------------------------
# Scale and visualize the embedding vectors in 3D plots       
def plot_embedding_3D(X, tile, title=None):        
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    
    # Establish the indices for plotting as slices of the X matrix
    # Only need the foci upper index, all others can be sliced using the dimensions already stored
    foci_upper_index = wt_samplesize + foci_samplesize
    
    ax = fig.add_subplot(1, 2, tile, projection='3d')
    
    # Plot WT
    ax.plot(X[:wt_samplesize, 0], X[:wt_samplesize, 1], X[:wt_samplesize, 2], 'o', c="r", label='Wild Type', ls='None')
    # Plot Foci
    ax.plot(X[wt_samplesize:foci_upper_index, 0], X[wt_samplesize:foci_upper_index, 1], X[wt_samplesize:foci_upper_index, 2], 'o', c="b", label='Foci', ls='None')
    # Plot Nonround nuclei
    ax.plot(X[foci_upper_index:, 0], X[foci_upper_index:, 1], X[foci_upper_index:, 2], 'o', c="g", label='Non-round Nuclei', ls='None')    
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')    
    
    ax.legend(loc = 'upper left')
    
    

#----------------------------------------------------------------------
# Isomap projection 
print "Computing Isomap embedding"
t0 = time()
D_iso = manifold.Isomap(n_neighbors, n_components=3).fit_transform(D_scaled)
print "Done in time %.2fs " % (time() - t0)

#----------------------------------------------------------------------
# Locally linear embedding 
print "Computing LLE embedding"
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=3,
                                      method='standard')
t0 = time()
D_lle = clf.fit_transform(D_scaled)
print "Done in time %.2fs " % (time() - t0)
print "Reconstruction error: %g" % clf.reconstruction_error_

if opts.dimension == 2:
    pl.figure(figsize=(9,8),dpi=200)
    plot_embedding(D_iso, 1, "Isomap projection")
    plot_embedding(D_lle, 2,"Locally Linear Embedding")
    pl.subplots_adjust(left=None, bottom=None, right=None, wspace=0.15, hspace=None)
    #pl.savefig("manifold_fig.pdf",format="pdf",dpi=200, orientation='landscape', pad_inches=0)
    pl.show()    
else:
    # Twice as wide as it is tall.
    fig = plt.figure(figsize=plt.figaspect(0.5))    
    plot_embedding_3D(D_iso, 1, "Isomap projection")
    plot_embedding_3D(D_lle, 2, "Local Linear Embedding")
    plt.savefig("manifold_fig_3D.pdf",format="pdf",dpi=200, orientation='landscape', pad_inches=0)
    plt.show()



# close the data file
datafile.close()