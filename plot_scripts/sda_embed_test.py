from sklearn import metrics
from sklearn.cluster import KMeans

import logging, os, re
from optparse import OptionParser
import sys

import numpy as np
from tables import *
from extract_datasets import extract_labeled_chunkrange, extract_unlabeled_chunkrange

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# Parse commandline arguments
op = OptionParser()
op.add_option("--h5file",
              dest="inputfile", help="Read data labels from this hdf5 file.")
op.add_option("--reducedbasedir",
              dest="basedir", help="Base input directory for all reduced data files.")
op.add_option("--reducedfile",
              dest="reducedfile", help="Read the reduced data from this file.")
op.add_option("--size",
              dest="size", type="int", help="Use this many chunks of labeled data for the test.")
op.add_option("--iters",
              dest="iters", type="int", help="Do this many iterations of k-means clustering.")
op.add_option("--outputdir",
              dest="outputdir", help="Write output matrix to the specified directory.")


(opts, args) = op.parse_args()

np.random.seed(0)

###############################################################################
# Load a training set from the given .h5 file
datafile = openFile(opts.inputfile, mode = "r", title = "Data is stored here")

# Load the reduced data from the base + name
reduced_file_path = os.path.join(opts.basedir,opts.reducedfile)
#debug
print "read from: " + reduced_file_path
reduced_data_file = openFile(reduced_file_path, mode = "r", title = "SdA reduced data stored here")
arrays_list = reduced_data_file.listNodes("/recarrays", classname='Array')

# Extract model name
model_regex = re.compile("reduce_SdA.([\d_]+).*")  
match = model_regex.match(opts.reducedfile)
if match is not None:    
        this_model = match.groups()[0]
        # debug
        print "Extracted model name: " + this_model
else:
        print "Could not extract model name from this file"
        sys.exit()


# Load the reduced data from a different file
X_reduced = extract_unlabeled_chunkrange(reduced_data_file, opts.size)
# Extract some of the dataset from the datafile
X_ignore, labels = extract_labeled_chunkrange(datafile, opts.size)

true_k =  np.unique(labels[:,0]).shape[0]

# done, close h5 files
datafile.close()
reduced_data_file.close()

# make sure we're dealing with the same number of labels as data points
num_pts = labels.shape[0]
X_data = X_reduced[0:num_pts,:]
assert(labels.shape[0] == X_data.shape[0])

#debug
print "Got here successfully, should be good to go"
sys.exit()

###############################################################################

# Build the output array
SdA_results = np.zeros((1,opts.iters))

    
for j in range(0,opts.iters,1):
    km = KMeans(k=true_k, init='k-means++', max_iter=1000, n_init=10, verbose=1)         
    km.fit(X_data)
    
    print "Homogeneity: %0.3f" % metrics.homogeneity_score(labels[:,0], km.labels_)        
    SdA_results[0,j] = metrics.homogeneity_score(labels[:,0], km.labels_)
        

# Save the data to a file
outfile = os.path.join(opts.outputdir,this_model + ".npy")
f = open(outfile, 'w')
np.save(f, SdA_results)
f.close()