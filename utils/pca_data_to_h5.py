from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import logging, os
from optparse import OptionParser
import sys

import numpy as np
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
              dest="outputfile", help="Write output matrices to .h5 output file.")
op.add_option("--size",
              dest="size", type="int", help="Extract the first size chunks of the data set and labels.")

(opts, args) = op.parse_args()


np.random.seed(0)

###############################################################################
# Load a training set from the given .h5 file
datafile = openFile(opts.inputfile, mode = "r", title = "Data is stored here")

# Extract some of the dataset from the datafile
X, labels = extract_labeled_chunkrange(datafile, opts.size)

# Remove the first four indexing features introduced by CP
X = X[:,5:-1]

# Open and prepare an hdf5 file 
filename = opts.outputfile
h5file = openFile(filename, mode = "w", title = "Reduced Data File")

# Create a new group under "/" (root)
arrays_group = h5file.createGroup("/", 'recarrays', 'The object data arrays')
zlib_filters = Filters(complib='zlib', complevel=5)

# done, close h5 files
datafile.close()

###############################################################################

pca = PCA(n_components=50)
D = scale(X[:,0:612])
pca.fit(D)
X_pca = pca.transform(D)


# For the specified number of principal components, do the clustering
for i in [10,20,30,40,50]:
    atom = Atom.from_dtype(X_pca.dtype)
    ds = h5file.createCArray(where=arrays_group, name=str(i), atom=atom, shape=X_pca.shape, filters=zlib_filters)
    ds[:] = X_pca[:,0:(i-1)]
    h5file.flush()     

h5file.close()
         