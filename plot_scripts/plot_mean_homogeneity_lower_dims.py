""" Plot the line graph with standard deviations of each of the algorithms for dimensionality reduction, as produced by the *_embed_test.py scripts """

import matplotlib as mpl
mpl.use('pdf')	# needed so that you can plot in a batch job with no X server (undefined $DISPLAY) problems

from matplotlib.offsetbox import TextArea, AnnotationBbox

import numpy as np
import pylab as P
import os
from optparse import OptionParser

op = OptionParser()
op.add_option("--pca",
              dest="pcainput", help="Read PCA data input from this file.")
op.add_option("--lle",
              dest="lleinput", help="Read LLE data input from this file.")
op.add_option("--iso",
              dest="isoinput", help="Read ISOMAP data input from this file.")
op.add_option("--sda",
              dest="sdainput", help="Read SdA models data input from this file.")
op.add_option("--output",
              dest="outfile", help="Write the pdf figure to this file.")
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
ax = fig.add_subplot(111)

x = np.arange(1,6,1)
ax.errorbar(x,pca_means,yerr=pca_std, elinewidth=2, capsize=3, label="PCA", lw=1.5, fmt='--o')
ax.errorbar(x,lle_means,yerr=lle_std, elinewidth=2, capsize=3, label="LLE", lw=1.5, fmt='--o')
ax.errorbar(x,isomap_means,yerr=isomap_std, elinewidth=2, capsize=3, label="ISOMAP", lw=1.5, fmt='--o')

# Add in the top 5 performing SdA models
#0: 1000_700_400_50 , 0.492122026616
#1: 700_500_300_50 , 0.43154219181
#2: 700_700_100_50 , 0.428561885442
#3: 1000_600_300_50 , 0.400669383048
#4: 700_900_100_50 , 0.39662856001
sda_top5 = []
top5 = ['1000_700_400_50','700_500_300_50','700_700_100_50','1000_600_300_50','700_900_100_50']
for model in top5:
    input_npy = os.path.join(opts.sdainput,model + ".npy")
    model_results = np.load(input_npy)
    sda_top5.append(np.mean(model_results))

ax.plot([1,1,1,1,1],sda_top5,'y*')


offsetbox = TextArea("SdA Top 5", minimumdescent=False)
xy = (1, 0.435)

ab = AnnotationBbox(offsetbox, xy,
                    xybox=(0.8, xy[1]),
                    xycoords='data',
                    boxcoords=("axes fraction", "data"),
                    box_alignment=(0.,0.5),
                    arrowprops=dict(arrowstyle="->"))
ax.add_artist(ab)

P.xlim(0,6)
P.ylim(0,0.45)
P.title('Average homogeneity for K = 3')
P.xlabel('Dimensions')
P.ylabel('Average Homogeneity')
locs, labels = P.xticks()   # get the xtick location and labels, re-order them so they match the experimental data
P.xticks(locs,['',50,40,30,20,10])
P.legend(loc = 3)    # legend lower left

P.savefig(opts.outfile, dpi=100, format="pdf")