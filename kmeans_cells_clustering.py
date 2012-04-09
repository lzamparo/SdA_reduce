"""
=======================================
Clustering cells using k-means
=======================================

This script (adapted from the scikit-learn k-means document clustering example)
is an attempt to cluster data points derived from cell images by phenotype. 

Two algorithms are demoed: ordinary k-means and its faster cousin minibatch
k-means.

"""

# Adapted from kmeans_document_clustering_example.py (http://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html#example-cluster-plot-mini-batch-kmeans-py) 
# Author: Lee Zamparo
#
# License: Simplified BSD

from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from tables import *

import logging, os
from optparse import OptionParser
import sys
from time import time

import numpy as np
import utils.extract_datasets

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm.")
op.add_option("--h5file",
              dest="inputfile", help="Read data input from this hdf5 file.")
op.add_option("--size",
              dest="size", type="int", help="Extract the first size chunks of the data set and labels.")

print __doc__
op.print_help()

(opts, args) = op.parse_args()


###############################################################################
# Load a training set from the given .h5 file
datafile = openFile(opts.inputfile, mode = "r", title = "Data is stored here")

# Extract some of the dataset from the datafile
dataset, labels = utils.extract_datasets.extract_datasets(datafile, opts.size)

true_k = np.unique(labels[:,0]).shape[0]


###############################################################################
# Do the actual clustering

if opts.minibatch:
    km = MiniBatchKMeans(k=true_k, init='k-means++', n_init=1,
                         init_size=1000,
                         batch_size=1000, verbose=1)
else:
    km = KMeans(k=true_k, init='k-means++', max_iter=100, n_init=1, verbose=1)

print "Clustering sparse data with %s" % km
t0 = time()
km.fit(dataset[:,2:-1])
print "done in %0.3fs" % (time() - t0)
print

print "Homogeneity: %0.3f" % metrics.homogeneity_score(labels[:,0], km.labels_)
print "Completeness: %0.3f" % metrics.completeness_score(labels[:,0], km.labels_)
print "V-measure: %0.3f" % metrics.v_measure_score(labels[:,0], km.labels_)
print "Adjusted Rand-Index: %.3f" % \
    metrics.adjusted_rand_score(labels[:,0], km.labels_)
print "Silhouette Coefficient: %0.3f" % metrics.silhouette_score(
    dataset[:,2:-1], labels[:,0], sample_size=1000)

# done, close h5 files
datafile.close()