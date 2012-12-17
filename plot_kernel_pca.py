"""
==========
Kernel PCA
==========

This script performs two PCA eigenvector decompositions (one regular PCA, one Kernel PCA)
and plots the data projected onto the first two principal components, in at attempt to 
evaluate the differences between the two algorithms on a selection from an image screen
data set.  

Adapted from the example provided by Mathieu Blondel: http://scikit-learn.org/stable/_downloads/plot_kernel_pca.py

"""
print __doc__

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tables import *

import logging
from optparse import OptionParser
import sys

import utils.extract_datasets
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import scale;

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
(opts, args) = op.parse_args()

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
D_scaler = scale(D)

kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=0.5)
D_kpca = kpca.fit_transform(D_scaler)
D_back = kpca.inverse_transform(D_kpca)
pca = PCA()
D_pca = pca.fit_transform(D_scaler)

# Plot results.  Manipulate D_scaled by standardizing the PCA matrix to shrink
# plotting axis if required.
x_min, x_max = np.min(D_pca, 0), np.max(D_pca, 0)
D_scaled = D_pca
#D_scaled = (D_pca - x_min) / (x_max - x_min)

pl.figure(figsize=(10,7),dpi=400)
pl.subplot(1, 2, 1, aspect='equal')

# Establish the indices for plotting as slices of the X matrix
# Only need the foci upper index, all others can be sliced using the dimensions already stored
foci_upper_index = wt_samplesize + foci_samplesize

pl.plot(D_scaled[:wt_samplesize, 0], D_scaled[:wt_samplesize, 1], "ro")
pl.plot(D_scaled[wt_samplesize:foci_upper_index, 0], D_scaled[wt_samplesize:foci_upper_index, 1], "bo")
pl.plot(D_scaled[foci_upper_index:, 0], D_scaled[foci_upper_index:, 1], "go")
pl.title("Projection by PCA")
pl.xlabel("1st principal component")
pl.ylabel("2nd component")

pl.subplot(1, 2, 2, aspect='equal')
pl.plot(D_kpca[:wt_samplesize, 0], D_kpca[:wt_samplesize, 1], "ro")
pl.plot(D_kpca[wt_samplesize:foci_upper_index, 0], D_kpca[wt_samplesize:foci_upper_index, 1], "bo")
pl.plot(D_kpca[foci_upper_index:, 0], D_kpca[foci_upper_index:, 1], "go")
pl.title("Projection by KPCA")
pl.xlabel("1st principal component in space induced by $\phi$")
pl.ylabel("2nd component")
pl.legend( ('Wild Type', 'Foci', 'Non-round Nuclei'), loc="upper right")

pl.subplots_adjust(0.02, 0.10, 0.98, 0.94, 0.15, 0.15)
pl.savefig("pcafig_2D.pdf",format="pdf",dpi=200)

#  Now generate a 3D pca figure with same data
fig = plt.figure(figsize=plt.figaspect(0.5),dpi=400)

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot(D_scaled[:wt_samplesize, 0], D_scaled[:wt_samplesize, 1], D_scaled[:wt_samplesize, 2], 'o', c="r", label='Wild Type', ls='None')
ax.plot(D_scaled[wt_samplesize:foci_upper_index, 0], D_scaled[wt_samplesize:foci_upper_index, 1], D_scaled[wt_samplesize:foci_upper_index, 2], 'o', c="b", label='Foci', ls='None')
ax.plot(D_scaled[foci_upper_index:, 0], D_scaled[foci_upper_index:, 1], D_scaled[foci_upper_index:, 2], 'o', c="g", label='Non-round Nuclei', ls='None')

ax.set_xlabel('1st principal component')
ax.set_ylabel('2nd component')
ax.set_zlabel('3rd component')    

ax.set_title("Projection by PCA")
ax.legend(loc = 'upper left')

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot(D_kpca[:wt_samplesize, 0], D_kpca[:wt_samplesize, 1], D_kpca[:wt_samplesize, 2],'o', c="r", label='Wild Type', ls='None')
ax.plot(D_kpca[wt_samplesize:foci_upper_index, 0], D_kpca[wt_samplesize:foci_upper_index, 1], D_kpca[wt_samplesize:foci_upper_index, 2], 'o', c="b", label='Foci', ls='None')
ax.plot(D_kpca[foci_upper_index:, 0], D_kpca[foci_upper_index:, 1], D_kpca[foci_upper_index:, 2], 'o', c="g", label='Non-round Nuclei', ls='None')

ax.set_xlabel('1st principal component in space induced by $\phi$')
ax.set_ylabel('2nd component')
ax.set_zlabel('3rd component')    

ax.set_title("Projection by KPCA")
ax.legend(loc = 'upper left')

plt.savefig("pcafig_3D.pdf",format="pdf",dpi=200)

# close the data file
datafile.close()