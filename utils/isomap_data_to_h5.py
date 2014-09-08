from sklearn.preprocessing import scale
from sklearn.manifold import Isomap

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
              dest="outputfile", help="Write matrices to .h5 output file.")
op.add_option("--size",
              dest="size", type="int", help="Extract the first size chunks of the data set and labels.")

(opts, args) = op.parse_args()


np.random.seed(0)

###############################################################################
# Load a training set from the given .h5 file
datafile = openFile(opts.inputfile, mode = "r", title = "Data is stored here")

# Open and prepare an hdf5 file 
filename = opts.outputfile
h5file = openFile(filename, mode = "w", title = "Reduced Data File")

# Create a new group under "/" (root)
arrays_group = h5file.createGroup("/", 'recarrays', 'The object data arrays')
zlib_filters = Filters(complib='zlib', complevel=5)

# Extract some of the dataset from the datafile
X, labels = extract_labeled_chunkrange(datafile, opts.size)

# Remove the first four indexing features introduced by CP
X = X[:,5:-1]

# done, close h5 files
datafile.close()

###############################################################################

D = scale(X[:,0:612])

n_samples, n_features = D.shape
# chosen by hyperparam search in a separate test.
n_neighbors = 10

name_dict = {10: 'dim10', 20: 'dim20', 30: 'dim30', 40: 'dim40', 50: 'dim50'}

for i in [10,20,30,40,50]: 
    isomap = Isomap(n_neighbors, n_components=i)
    X_iso = isomap.fit_transform(D)
    atom = Atom.from_dtype(X_iso.dtype)
    ds = h5file.createCArray(where=arrays_group, name=name_dict[i], atom=atom, shape=X_iso.shape, filters=zlib_filters)
    ds[:] = X_iso
    h5file.flush()    
        
h5file.close()


# Save the data to a DataFrame
#data = isomap_gmm_results.ravel()
#dims = ['10' for i in xrange(opts.iters)]
#dims.extend(['20' for i in xrange(opts.iters)])
#dims.extend(['30' for i in xrange(opts.iters)])
#dims.extend(['40' for i in xrange(opts.iters)])
#dims.extend(['50' for i in xrange(opts.iters)])
#method = ['isomap' for i in xrange(len(dimension_list) * int(opts.iters))]
#results_df = pd.DataFrame({"data": data, "dimension": dims, "method": method})
#results_df.to_csv(opts.outputfile+"_df.csv", index=False)
