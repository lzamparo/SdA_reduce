""" Plot the line graph with standard deviations of each of the algorithms for dimensionality reduction, as produced by the *_embed_test.py scripts 

Writes two pdf files to the outfile prefix: .gmm.pdf and .kmeans.pdf, depending on the clustering model used to measure homogeneity.
"""

import matplotlib as mpl
mpl.use('pdf')	# needed so that you can plot in a batch job with no X server (undefined $DISPLAY) problems

from matplotlib.offsetbox import TextArea, AnnotationBbox

import numpy as np
import pylab as P
import os
from optparse import OptionParser
from parse_homogeneity_results import return_top, parse_dir

op = OptionParser()
op.add_option("--pca",
              dest="pcainput", help="Read PCA data input from this file.")
op.add_option("--lle",
              dest="lleinput", help="Read LLE data input from this file.")
op.add_option("--iso",
              dest="isoinput", help="Read ISOMAP data input from this file.")
op.add_option("--sda",
              dest="sdainput", help="Read SdA models data input below this directory.")
op.add_option("--output",
              dest="outfile", help="Write the pdf figures to this file prefix.")
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

# Compare the top n models for SdA against PCA, LLE, ISOMAP
n = 5

####################  GMM test ####################

# Plot the mean for each results matrix with standard deviation bars
fig = P.figure()
ax = fig.add_subplot(111)

x = np.arange(1,6,1)
ax.errorbar(x,pca_means,yerr=pca_std, elinewidth=2, capsize=3, label="PCA", lw=1.5, fmt='--o')
ax.errorbar(x,lle_means,yerr=lle_std, elinewidth=2, capsize=3, label="LLE", lw=1.5, fmt='--o')
ax.errorbar(x,isomap_means,yerr=isomap_std, elinewidth=2, capsize=3, label="ISOMAP", lw=1.5, fmt='--o')

# read all dimension folders below opts.sdainput
dims_list = os.listdir(opts.sdainput)
dims_list = [i for i in dims_list if i.endswith('0')]

# dive in and extract the top n models (n = 5) from each dim
# fill the sda_results with list of lists, each sub-list representing the points on the y-axis to plot (homogeneity results)
sda_results = []

for dim, i in enumerate(dims_list):
    parsed_vals = parse_dir(os.path.join(opts.sdainput,str(dim),'gmm'))
    results_dict = return_top(parsed_vals,n)
    sda_results.append(results_dict.values())
    x_vals = np.ones((n,),dtype=np.int) * (i + 1)
    ax.plot(x_vals.tolist(),sda_results[i],'y*',label="SdA",markersize=10)

P.xlim(0,6)
P.ylim(0,0.50)
P.title('3-component Gaussian mixture model test')
P.xlabel('Dimension of the Data')
P.ylabel('Average Homogeneity')
locs, labels = P.xticks()   # get the xtick location and labels, re-order them so they match the experimental data
P.xticks(locs,['',50,40,30,20,10])
P.legend(loc = 7,numpoints=1)    # legend centre right

outfile = opts.outfile + ".gmm.pdf"
P.savefig(outfile, dpi=100, format="pdf")

####################  K-means test ####################

fig = P.figure()
ax = fig.add_subplot(111)

x = np.arange(1,6,1)
ax.errorbar(x,pca_means,yerr=pca_std, elinewidth=2, capsize=3, label="PCA", lw=1.5, fmt='--o')
ax.errorbar(x,lle_means,yerr=lle_std, elinewidth=2, capsize=3, label="LLE", lw=1.5, fmt='--o')
ax.errorbar(x,isomap_means,yerr=isomap_std, elinewidth=2, capsize=3, label="ISOMAP", lw=1.5, fmt='--o')

# read all dimension folders below opts.sdainput
dims_list = os.listdir(opts.sdainput)
dims_list = [i for i in dims_list if i.endswith('0')]

# dive in and extract the top n models (n = 20) from each dim
# fill the sda_results with list of lists, each sub-list representing the points on the y-axis to plot (homogeneity results)
sda_results = []

for dim, i in enumerate(dims_list):
    parsed_vals = parse_dir(os.path.join(opts.sdainput,str(dim),'kmeans'))
    results_dict = return_top(parsed_vals,n)
    sda_results.append(results_dict.values())
    x_vals = np.ones((n,),dtype=np.int) * (i + 1)
    ax.plot(x_vals.tolist(),sda_results[i],'y*',label="SdA",markersize=10)

P.xlim(0,6)
P.ylim(0,0.50)
P.title('3-component K-means mixture model test')
P.xlabel('Dimension of the Data')
P.ylabel('Average Homogeneity')
locs, labels = P.xticks()   # get the xtick location and labels, re-order them so they match the experimental data
P.xticks(locs,['',50,40,30,20,10])
P.legend(loc = 7,numpoints=1)    # legend centre right
outfile = opts.outfile + ".kmeans.pdf"
P.savefig(opts.outfile, dpi=100, format="pdf")