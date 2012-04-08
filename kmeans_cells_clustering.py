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

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np


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

print __doc__
op.print_help()

(opts, args) = op.parse_args()
if len(args) > 1:
    op.error("This script only takes one argument, the h5 file.")
    sys.exit(1)


###############################################################################
# Load a training set from the given .h5 file
datafile = openFile(opts.inputfile, mode = "r", title = "Data is stored here")

# Draw the data set from here


# Uncomment the following to do the analysis on all the categories
#categories = None

print "Loading 20 newsgroups dataset for categories:"
print categories

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

print "%d documents" % len(dataset.data)
print "%d categories" % len(dataset.target_names)
print

labels = dataset.target
true_k = np.unique(labels).shape[0]

print "Extracting features from the training dataset using a sparse vectorizer"
t0 = time()
vectorizer = Vectorizer(max_df=0.95, max_features=10000)
X = vectorizer.fit_transform(dataset.data)

print "done in %fs" % (time() - t0)
print "n_samples: %d, n_features: %d" % X.shape
print


###############################################################################
# Do the actual clustering

if opts.minibatch:
    km = MiniBatchKMeans(k=true_k, init='k-means++', n_init=1,
                         init_size=1000,
                         batch_size=1000, verbose=1)
else:
    km = KMeans(k=true_k, init='random', max_iter=100, n_init=1, verbose=1)

print "Clustering sparse data with %s" % km
t0 = time()
km.fit(X)
print "done in %0.3fs" % (time() - t0)
print

print "Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_)
print "Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_)
print "V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_)
print "Adjusted Rand-Index: %.3f" % \
    metrics.adjusted_rand_score(labels, km.labels_)
print "Silhouette Coefficient: %0.3f" % metrics.silhouette_score(
    X, labels, sample_size=1000)

print
