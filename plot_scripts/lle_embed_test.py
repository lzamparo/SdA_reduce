from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.manifold import LocallyLinearEmbedding

import logging, os
from optparse import OptionParser
import sys
from time import time

import numpy as np
from tables import *
from utils.extract_datasets import extract_labeled_chunkrange


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--h5file",
              dest="inputfile", help="Read data input from this hdf5 file.")
op.add_option("--output",
              dest="outputfile", help="Write matrix to .npy output file.")
op.add_option("--size",
              dest="size", type="int", help="Extract the first size chunks of the data set and labels.")
op.add_option("--high",
              dest="high", type="int", help="Start at high dimensions.")
op.add_option("--low",
              dest="low", type="int", help="End at low dimensions.")
op.add_option("--step",
              dest="step", type="int", help="Go from high to low by step sizes")
op.add_option("--iters",
              dest="iters", type="int", help="Do this many iterations of k-means clustering")

(opts, args) = op.parse_args()


np.random.seed(0)

###############################################################################
# Load a training set from the given .h5 file
datafile = openFile(opts.inputfile, mode = "r", title = "Data is stored here")

# Extract some of the dataset from the datafile
X, labels = extract_labeled_chunkrange(datafile, opts.size)

# Remove the first four indexing features introduced by CP
X = X[:,5:-1]

true_k =  np.unique(labels[:,0]).shape[0]

# done, close h5 files
datafile.close()

###############################################################################

# Build the output arrays
cells = opts.high / opts.step
lle_results = np.zeros((cells,opts.iters))

D = scale(X[:,0:612])

n_samples, n_features = D.shape
n_neighbors = 20

# For the specified number of principal components, do the clustering
dimension_list = range(opts.low, opts.high + 1, opts.step)
for i in dimension_list:
    index = (i / opts.step) - 1     
    clf = LocallyLinearEmbedding(n_neighbors, n_components=i,
                                          method='standard')
    X_lle = clf.fit_transform(D)
    
    
    for j in range(0,opts.iters,1):
        km = KMeans(k=true_k, init='k-means++', max_iter=1000, n_init=10, verbose=1)  
        #print "Clustering LLE data with %s" % km
        
        km.fit(X_lle)
        print "Homogeneity: %0.3f" % metrics.homogeneity_score(labels[:,0], km.labels_)      
        lle_results[index,j] = metrics.homogeneity_score(labels[:,0], km.labels_)
        
# Take the mean across runs?
#lle_means = lle_results.mean(axis=1)

# Save the data to a file:
#np.savetxt("lle_results.txt", lle_means)
np.save(opts.outputfile, lle_results)