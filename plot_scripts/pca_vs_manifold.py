from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing, manifold
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
op.add_option("--h5file",
              dest="inputfile", help="Read data input from this hdf5 file.")
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

print __doc__

(opts, args) = op.parse_args()


np.random.seed(0)

###############################################################################
# Load a training set from the given .h5 file
datafile = openFile(opts.inputfile, mode = "r", title = "Data is stored here")

# Extract some of the dataset from the datafile
X, labels = utils.extract_datasets.extract_datasets(datafile, opts.size)

# Remove the first four indexing features introduced by CP
X = X[:,5:-1]

true_k =  np.unique(labels[:,0]).shape[0]

# done, close h5 files
datafile.close()

###############################################################################

# Build the output arrays
cells = opts.high / opts.step
pca_results = np.zeros((cells,opts.iters))
#isomap_results = np.zeros((cells,opts.iters))
#lle_results = np.zeros((cells,opts.iters))

pca = PCA(n_components=opts.high)
pca.fit(X[:,0:612])
X_pca = pca.transform(X[:,0:612])

#X_scaled = preprocessing.scale(X[:,0:612])

#n_samples, n_features = X_scaled.shape
#n_neighbors = 20

# For the specified number of principal components, do the clustering
dimension_list = range(opts.low, opts.high + 1, opts.step)
for i in dimension_list:
    index = (i / opts.step) - 1 
    
    #print "Computing Isomap embedding"
    #t0 = time()
    #X_iso = manifold.Isomap(n_neighbors, n_components=i).fit_transform(X_scaled)
    #print "Done in time %.2fs " % (time() - t0)    

    #print "Computing LLE embedding"
    #clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=i,
                                          #method='standard')
    #t0 = time()
    #X_lle = clf.fit_transform(X_scaled)
    #print "Done in time %.2fs " % (time() - t0)
    #print "Reconstruction error: %g" % clf.reconstruction_error_    
    
    for j in range(0,opts.iters,1):
        km = KMeans(k=true_k, init='k-means++', max_iter=1000, n_init=10, verbose=1)  
        #print "Clustering Isomap data with %s" % km
        t0 = time()        
        km.fit(X_pca[:,0:(i-1)])
        #km.fit(X_iso)
        print "done in %0.3fs" % (time() - t0)
        print
        
        print "Homogeneity: %0.3f" % metrics.homogeneity_score(labels[:,0], km.labels_)        
        pca_results[index,j] = metrics.homogeneity_score(labels[:,0], km.labels_)
        
        #print "Homogeneity: %0.3f" % metrics.homogeneity_score(labels[:,0], km.labels_)        
        #isomap_results[index,j] = metrics.homogeneity_score(labels[:,0], km.labels_)
        
        #print "Clustering LLE data with %s" % km
        #t0 = time()        
        #km.fit(X_lle)
        #print "done in %0.3fs" % (time() - t0)
        #print        
        #lle_results[index,j] = metrics.homogeneity_score(labels[:,0], km.labels_)
        
# Take the mean across runs for the pca homogenaity
pca_means = pca_results.mean(axis=1)
#isomap_means = isomap_results.mean(axis=1)
#lle_means = lle_results.mean(axis=1)

# Save the data to a file:
np.savetxt("pca_results.txt", pca_means)
#np.savetxt("isomap_results.txt", isomap_means)
#np.savetxt("lle_results.txt", lle_means)