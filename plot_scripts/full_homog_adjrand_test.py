from sklearn import metrics
from sklearn.mixture import GMM
from sklearn.preprocessing import scale

from tables import *
from extract_datasets import extract_labeled_chunkrange
from time import time
import cPickle as pkl
import os

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

p.add_argument('-i', metavar='<inputfile>',
               default='/scratch/z/zhaolei/lzamparo/sm_rep1_data/sample.h5',
                help='input data')
args = p.parse_args()


datafile = openFile(args.i,'r')
X, labels = extract_labeled_chunkrange(datafile, 3)
X = X[:,5:-1]
import numpy as np
true_k =  np.unique(labels[:,0]).shape[0]
datafile.close()
D = scale(X)

gaussmix = GMM(n_components=true_k, covariance_type='tied', n_init=10, n_iter=1000)
t0 = time()
gaussmix.fit(D)
print "[Worker %d] GMM fit, took %.2fs" % (os.getpid(), time() - t0)
gaussmix_labels = gaussmix.predict(D)
print "[Worker %d] Homogeneity %0.3f" % (os.getpid(), metrics.homogeneity_score(labels[:,0], gaussmix_labels))
print "[Worker %d] Adjusted Rand Index %0.3f" % (os.getpid(), metrics.adjusted_rand_score(labels[:,0],gaussmix_labels))
    