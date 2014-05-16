"""
==========
Kernel PCA gamma parameter CV pipeline
==========

Use a pipeline to find the best value for gamma parameter in the RBF kernel for kernel PCA.

Adapted from: 
    http://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html#example-decomposition-plot-kernel-pca-py
    http://scikit-learn.org/stable/auto_examples/grid_search_digits.html#example-grid-search-digits-py

"""
import numpy as np
import pickle

from optparse import OptionParser
from tables import *

from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score, make_scorer
from extract_datasets import extract_labeled_chunkrange
from sklearn.preprocessing import scale
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


np.random.seed(0)

# parse commandline arguments
op = OptionParser()
op.add_option("--h5file",
              dest="inputfile", help="Read data input from this hdf5 file.")
op.add_option("--size",
              dest="size", type="int", help="Extract the first size chunks of the data set and labels.")
op.add_option("--sample-size",
              dest="samplesize", type="int", help="The max size of the samples")
op.add_option("--output",
              dest="outfile", help="Write the estimator model to this file.")
op.add_option("--num-jobs",
              dest="jobs", type="int", help="Use these number of jobs in parallel for GridSearchCV")

(opts, args) = op.parse_args()

###############################################################################
# Load a training set from the given .h5 file
datafile = openFile(opts.inputfile, mode = "r", title = "Data is stored here")

# Extract some of the dataset from the datafile
X, labels = extract_labeled_chunkrange(datafile, opts.size)

# Sample from the dataset
wt_points = np.nonzero(labels[:,0] == 0)[0]
foci_points = np.nonzero(labels[:,0] == 1)[0]
ab_nuclei_points = np.nonzero(labels[:,0] == 2)[0]

wt_data = X[wt_points,5:]
foci_data = X[foci_points,5:]
ab_nuclei_data = X[ab_nuclei_points,5:]

wt_labels = labels[wt_points,0]
foci_labels = labels[foci_points,0]
ab_nuclei_labels = labels[ab_nuclei_points,0]

# Figure out the sample sizes based on the shape of the *_labels arrays and the 
# sample size argument

wt_samplesize = min(opts.samplesize,wt_data.shape[0])
foci_samplesize = min(opts.samplesize,foci_data.shape[0])
ab_nuclei_samplesize = min(opts.samplesize, ab_nuclei_data.shape[0]) 

# Use np.random.permutation(array)[0:size,:] to sample u at random
# from the strata.
wt_data_sample = np.random.permutation(wt_data)[0:wt_samplesize,:]
foci_data_sample = np.random.permutation(foci_data)[0:foci_samplesize,:]
ab_nuclei_sample = np.random.permutation(ab_nuclei_data)[0:ab_nuclei_samplesize,:]

D = np.vstack((wt_data_sample,foci_data_sample,ab_nuclei_sample))
D_labels = np.hstack((wt_labels,foci_labels,ab_nuclei_labels))
D_scaled = scale(D)

datafile.close()

##################

# Set up the kPCA -> kmeans -> v-measure pipeline
kpca = KernelPCA(kernel="rbf")
kmeans = KMeans(n_clusters=3)
pipe = Pipeline(steps=[('kpca', kpca), ('kmeans', kmeans)])

# Range of parameters to consider for gamma in the RBF kernel for kPCA
gammas = np.logspace(-10,3,num=50)

# Make a scoring function for the pipeline
v_measure_scorer = make_scorer(v_measure_score)

# Set the kpca model parameters to cycle over using '__' a prefix
estimator = GridSearchCV(pipe, dict(kpca__gamma=gammas), scoring=v_measure_scorer, n_jobs=opts.jobs)
estimator.fit(D_scaled,D_labels)

# Dump the estimator to a file
f = file(opts.outfile, 'wb')
pickle.dump(estimator, f)
f.close()

# Report the best parameter values
print("Best estimator found on test data set:")
print()
print(estimator.best_estimator_)
print()
print("Best parameters fond on test data set:")
print()
print(estimator.best_params_)
print()
print("Grid scores on development set:")
print()
for params, mean_score, scores in estimator.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
        % (mean_score, scores.std() / 2, params))
print()





