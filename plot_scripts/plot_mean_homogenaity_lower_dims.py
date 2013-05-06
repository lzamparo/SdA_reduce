""" Plot the line graph with standard deviations of each of the algorithms for dimensionality reduction, as produced by the *_embed_test.py scripts """

import numpy as np
import pylab as P
from optparse import OptionParser

op = OptionParser()
op.add_option("--pca",
              dest="pcainput", help="Read PCA data input from this file.")
op.add_option("--lle",
              dest="lleinput", help="Read LLE data input from this file.")
op.add_option("--iso",
              dest="isoinput", help="Read ISOMAP data input from this file.")
op.add_option("--output",
              dest="outfile", help="Write the EPS figure to this file.")
(opts, args) = op.parse_args()

# Load the data matrices
pca = np.load(opts.pcainput)
lle = np.load(opts.lleinput)
isomap = np.load(opts.isoinput)

# Calculate the means and std devs of each 
pca_means = pca.mean(axis = 1)
pca_std = pca.std(axis = 1)
lle_means = lle.mean(axis = 1)
lle_std = lle.std(axis = 1)
isomap_means = isomap.mean(axis = 1)
isomap_std = isomap.std(axis = 1)


# Plot the mean for each results matrix with standard deviation bars
fig = P.figure()

x = np.arange(1,6,1)
P.errorbar(x,pca_means,yerr=pca_std, elinewidth=2, capsize=3, label="PCA", lw=1.5, fmt='--o')
P.errorbar(x,lle_means,yerr=lle_std, elinewidth=2, capsize=3, label="LLE", lw=1.5, fmt='--o')
P.errorbar(x,isomap_means,yerr=isomap_std, elinewidth=2, capsize=3, label="ISOMAP", lw=1.5, fmt='--o')

P.xlim(0,6)
P.ylim(0,0.45)
P.title('Average homogenaity for K = 3')
P.xlabel('Dimensions')
P.ylabel('Average Homogenaity')
locs, labels = P.xticks()   # get the xtick location and labels, re-order them so they match the experimental data
P.xticks(locs,['',50,40,30,20,10])
P.legend(loc = 3)    # legend lower left

P.savefig(opts.outfile, dpi=100, format="pdf")