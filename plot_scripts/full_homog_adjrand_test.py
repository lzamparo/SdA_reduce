from sklearn import metrics
from sklearn.mixture import GMM
from sklearn.preprocessing import scale

from tables import *
from extract_datasets import extract_labeled_chunkrange
from time import time
import cPickle as pkl

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

p.add_argument('-i', metavar='<inputfile>',
               default='/scratch/z/zhaolei/lzamparo/sm_rep1_data/sample.h5',
                help='input data')
p.add_argument('-o', metavar='<savefile>',
               default='/scratch/z/zhaolei/lzamparo/figures/full_dim_homog_test.pkl',
               help='load the dpmm from this pkl file')
args = p.parse_args()


datafile = open_file(args.i,'r')
X, labels = extract_labeled_chunkrange(datafile, 3)
X = X[:,5:-1]
import numpy as np
true_k =  np.unique(labels[:,0]).shape[0]
datafile.close()
D = scale(X)

full_dim_homog = []
full_dim_adjrand = []
time_stats = []

for j in xrange(10):
    gaussmix = GMM(n_components=true_k, covariance_type='tied', n_init=10, n_iter=1000)
    t0 = time()
    gaussmix.fit(D)
    print "GMM fit, took %.2fs" % (time() - t0)
    time_stats.append(time() - t0)
    gaussmix_labels = gaussmix.predict(D)
    print "Homogeneity %0.3f" % metrics.homogeneity_score(labels[:,0], gaussmix_labels)
    print "Adjusted Rand Index %0.3f" % metrics.adjusted_rand_score(labels[:,0],gaussmix_labels)
    full_dim_homog.append(metrics.homogeneity_score(labels[:,0], gaussmix_labels))
    full_dim_adjrand.append(metrics.adjusted_rand_score(labels[:,0],gaussmix_labels))
    
test_stats = {'time': time_stats, 'homogeneity': full_dim_homog, 'adjusted_rand_index': full_dim_adjrand}
pkl.dump(test_stats, open(args.o,'wb'), protocol=0)