"""
==========
Kernel PCA
==========

This script performs two PCA eigenvector decompositions (one regular PCA, one Kernel PCA)
and plots the data projected onto the first two principal components, in at attempt to 
evaluate the differences between the two algorithms on a selection from an image screen
data set.  

Adapted from the example provided by Mathieu Blondel

"""
print __doc__

import numpy as np
import pylab as pl
from tables import *

import logging
from optparse import OptionParser
import sys

import utils.extract_datasets
from sklearn.decomposition import PCA, KernelPCA

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

###############################################################################
# Load a training set from the given .h5 file
datafile = openFile(opts.inputfile, mode = "r", title = "Data is stored here")

# Extract some of the dataset from the datafile
X, labels = utils.extract_datasets.extract_datasets(datafile, opts.size)

# Sample from the dataset
wt_labels = np.nonzero(labels[:,0] == 0)[0]
foci_labels = np.nonzero(labels[:,0] == 1)[0]
ab_nuclei_labels = np.nonzero(labels[:,0] == 2)[0]

wt_data = X[wt_labels,5:]
foci_data = X[foci_labels,5:]
ab_nuclei_data = X[ab_nuclei_labels,5:]

# can just use np.random.permutation(array)[0:size,:] to sample u at random
# from the strata.


kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=0.5)
X_kpca = kpca.fit_transform(X)
X_back = kpca.inverse_transform(X_kpca)
pca = PCA()
X_pca = pca.fit_transform(X)

# Plot results

pl.figure()
pl.subplot(2, 2, 1, aspect='equal')
pl.title("Original space")
pl.plot(X[:200, 0], X[:200, 1], "ro")
pl.plot(X[200:, 0], X[200:, 1], "bo")
pl.xlabel("$x_1$")
pl.ylabel("$x_2$")

X1, X2 = np.meshgrid(np.linspace(-6, 6, 50), np.linspace(-6, 6, 50))
X_grid = np.array([np.ravel(X1), np.ravel(X2)]).T

# projection on the first principal component (in the phi space)
Z_grid = kpca.transform(X_grid)[:, 0].reshape(X1.shape)
pl.contour(X1, X2, Z_grid, colors='grey', linewidths=1, origin='lower')

pl.subplot(2, 2, 2, aspect='equal')
pl.plot(X_kpca[:200, 0], X_pca[:200, 1], "ro")
pl.plot(X_pca[200:, 0], X_pca[200:, 1], "bo")
pl.title("Projection by PCA")
pl.xlabel("1st principal component")
pl.ylabel("2nd component")

pl.subplot(2, 2, 3, aspect='equal')
pl.plot(X_kpca[:200, 0], X_kpca[:200, 1], "ro")
pl.plot(X_kpca[200:, 0], X_kpca[200:, 1], "bo")
pl.title("Projection by KPCA")
pl.xlabel("1st principal component in space induced by $\phi$")
pl.ylabel("2nd component")

pl.subplot(2, 2, 4, aspect='equal')
pl.plot(X_back[:200, 0], X_back[:200, 1], "ro")
pl.plot(X_back[200:, 0], X_back[200:, 1], "bo")
pl.title("Original space after inverse transform")
pl.xlabel("$x_1$")
pl.ylabel("$x_2$")

pl.subplots_adjust(0.02, 0.10, 0.98, 0.94, 0.04, 0.35)

pl.show()
