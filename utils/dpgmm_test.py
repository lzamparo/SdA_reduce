from sklearn.mixture import DPGMM
import sys
import logging, os
from optparse import OptionParser

import numpy as np
from tables import *
from extract_datasets import extract_unlabeled_chunkrange, extract_chunk_sizes

import contextlib,time
@contextlib.contextmanager
def timeit():
  t=time.time()
  yield
  print(time.time()-t,"sec")

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# Parse commandline arguments
op = OptionParser()
op.add_option("--h5file",
              dest="inputfile", help="Read unlabeled data from this hdf5 file.")
op.add_option("--size",
              dest="size", type="int", help="Use this many chunks of labeled data for the test.")

(opts, args) = op.parse_args()

np.random.seed(0)

# there is a max of 211 chunks of data in sm_rep1, so cap opts.size to this
if opts.size > 211:
  opts.size = 211

###############################################################################
# The unlabeled data h5 file
unlabeled_datafile = openFile(opts.inputfile, mode = "r")

# The labeled data h5 file
#labeled_datafile = openFile(labeldata, mode = "r")

# Load the reduced data from a different file
X_unlabeled = extract_unlabeled_chunkrange(unlabeled_datafile, opts.size)
chunk_sizes = extract_chunk_sizes(unlabeled_datafile)

# Extract some of the dataset from the datafile
# X_labeled, labels = extract_labeled_chunkrange(labeled_datafile, opts.size)

# done, close h5 files
#labeled_datafile.close()
unlabeled_datafile.close()

for chunks in np.arange(1, opts.size, step = 3):
  # Sample the specified number of points from X_unlabeled
  size = np.cumsum(chunk_sizes[:chunks])[-1]
  
  # Fit a Dirichlet process mixture of Gaussians using up to  ten components
  dpgmm = DPGMM(n_components=10, alpha=10.0, covariance_type='full')
  indices = np.arange(X_unlabeled.shape[0])
  np.random.shuffle(indices)
  X = X_unlabeled[indices[:size],]
  
  print("fitting a model with", size, "data points")
  with timeit():
    dpgmm.fit(X)
  print("Done!")
  print("AIC for this model & data: ", dpgmm.aic(X))
  print("BIC for this model & data: ", dpgmm.bic(X))
  Y_hat = dpgmm.predict(X)
  print ("Model assigned points to", np.max(Y_hat), "components")
  

# How can I best check this out? 
#color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])
#for i, (clf, title) in enumerate([(gmm, 'GMM'),
                                  #(dpgmm, 'Dirichlet Process GMM')]):
    #splot = plt.subplot(2, 1, 1 + i)
    #Y_ = clf.predict(X)
    #for i, (mean, covar, color) in enumerate(zip(
            #clf.means_, clf._get_covars(), color_iter)):
        #v, w = linalg.eigh(covar)
        #u = w[0] / linalg.norm(w[0])
        ## as the DP will not use every component it has access to
        ## unless it needs it, we shouldn't plot the redundant
        ## components.
        #if not np.any(Y_ == i):
            #continue
        #plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        ## Plot an ellipse to show the Gaussian component
        #angle = np.arctan(u[1] / u[0])
        #angle = 180 * angle / np.pi  # convert to degrees
        #ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        #ell.set_clip_box(splot.bbox)
        #ell.set_alpha(0.5)
        #splot.add_artist(ell)

    #plt.xlim(-10, 10)
    #plt.ylim(-3, 6)
    #plt.xticks(())
    #plt.yticks(())
    #plt.title(title)

#plt.show()