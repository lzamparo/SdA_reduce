from sklearn import metrics
from sklearn.mixture import GMM
from sklearn.preprocessing import scale
from sklearn.manifold import Isomap

import logging, os
from optparse import OptionParser
import sys
from time import time

import numpy as np
import pandas as pd
from tables import *
from extract_datasets import extract_labeled_chunkrange

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
isomap_gmm_results = np.zeros((cells,opts.iters))

D = scale(X[:,0:612])

n_samples, n_features = D.shape
# chosen by hyperparam search in a separate test.
n_neighbors = 10

# For the specified number of principal components, do the clustering
dimension_list = range(opts.low, opts.high + 1, opts.step)
for i in dimension_list:
    index = (i / opts.step) - 1 
    isomap = Isomap(n_neighbors, n_components=i)
    X_iso = isomap.fit_transform(D)
     
    for j in range(0,opts.iters,1): 
        gaussmix = GMM(n_components=true_k, covariance_type='tied', n_init=10, n_iter=1000)
        gaussmix.fit(X_iso)
        gaussmix_labels = gaussmix.predict(X_iso)
               
        print "Homogeneity: %0.3f" % metrics.homogeneity_score(labels[:,0], gaussmix_labels)
        isomap_gmm_results[index,j] = metrics.homogeneity_score(labels[:,0], gaussmix_labels)
        

# Save the data to a file:
np.save(opts.outputfile+ "_gmm",isomap_gmm_results)

# Save the data to a DataFrame
data = isomap_gmm_results.ravel()
dims = ['10' for i in xrange(opts.iters)]
dims.extend(['20' for i in xrange(opts.iters)])
dims.extend(['30' for i in xrange(opts.iters)])
dims.extend(['40' for i in xrange(opts.iters)])
dims.extend(['50' for i in xrange(opts.iters)])
method = ['isomap' for i in xrange(len(dimension_list) * int(opts.iters))]
results_df = pd.DataFrame({"data": data, "dimension": dims, "method": method})
results_df.to_csv(opts.outputfile+"_df.csv", index=False)
