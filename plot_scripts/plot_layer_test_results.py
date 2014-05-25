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
op.add_option("--topmodels",
              dest="topX", type=int, default=5, help="Compare the top # models from 3,5 layer families.")
op.add_option("--threedir",
              dest="threedir", help="3 layer input directories lie below here")
op.add_option("--fivedir",
              dest="fivedir", help="5 layer input directories lie below here")
op.add_option("--upperlimit",
              dest="upperlimit", default=50, type=int, help="Don't plot points in higher dimensions than this limit")
op.add_option("--compareunits",
              dest="makeunits", default=True, help="Produce a ReLU versus GB figure.  Produces two plots (3 layer, 5 layer)")
op.add_option("--comparelayers",
              dest="makelayers", default=True, help="Produce a 3 vs 5 layer figure.  Produces one plot (using GB units)")
op.add_option("--output",
              dest="outfile", help="Write the pdf figures to this file prefix.")
(opts, args) = op.parse_args()

# Make sure threedir, fivedir paths exist
if not os.path.exists(opts.fivedir) and os.path.exists(opts.threedir):
    print()
    print("Error: both 3,5 layer model dirs must be specified.")
    exit()

if opts.makeunits:

    # Plot the mean for each results matrix with standard deviation bars
    fig = P.figure()
    
    ax = fig.add_subplot(211)
    make_units_plot(ax, opts.topX, opts.threedir, "3 layer model embedding test", opts.upperlimit)      
    
    ax = fig.add_subplot(221)    
    make_units_plot(ax, opts.topX, opts.fivedir, "5 layer model embedding test", opts.upperlimit)     

    outfile = opts.outfile + ".gmm.pdf"
    P.savefig(outfile, dpi=100, format="pdf")
    
if opts.makelayers:
    
    # Plot the mean for each results matrix with standard deviation bars
    fig = P.figure()
    
    ax = fig.add_subplot(111)
    make_units_plot(opts.topX, opts.threedir, "3 layer vs 5 layer models")
    
    ax = fig.add_subplot(221)    
    make_units_plot(opts.topX, opts.threedir, "3 layer model embedding test")     
    
    outfile = opts.outfile + ".gmm.pdf"
    P.savefig(outfile, dpi=100, format="pdf")    
    


def make_units_plot(ax, topX, base_dir, title, limit):
    """ Make a ReLU vs GB networks plot
    
    :param: ax axis-object
    :param: ax Matplotlib sub-plot object 
    
    :param: topX int
    :param: topX use this number of top scoring models in the plot
    
    :param: base_dir string
    :param: base_dir read all input from below here
    
    :param: title string
    :param: title use this for the sub-figure title
    
    :param: limit int
    :param: limit don't plot data above this upper limit in dimension size
    """    
    
    # Plot the results from the top X models in the comparison figures
    n = topX
    
    # read all dimension folders below opts.threedir, opts.fivedir
    # plot the points in order of highest to lowest dimension, hence the sort & reverse.
    dims_list = os.listdir(base_dir)
    dims_list = [i for i in dims_list if i.endswith('0') and int(i) <= limit]
    dims_list.sort()
    dims_list.reverse()    
    
    x = np.arange(1,len(dims_list),1)
    
    # dive in and extract the top n models from each dim and each unit type
    # fill with list of lists, each sub-list representing the points on the y-axis to plot (homogeneity results)
    sda_results_relu = []
    sda_results_gb = []
        
    for i, dim in enumerate(dims_list):
        
        # extract values for GB .npy files
        parsed_gb_mean,parsed_gb_std = parse_dir(os.path.join(base_dir,str(dim),'gb'))
        # extract values for ReLU .npy files
        parsed_relu_mean,parsed_relu_std = parse_dir(os.path.join(base_dir,str(dim),'relu'))
        top_gb_mean = return_top(parsed_gb_mean,n)
        #top_gb_std = return_top(parsed_gb_std,n)
        top_relu_mean = return_top(parsed_relu_mean,n)
        #top_relu_std = return_top(parsed_relu_std,n)        
        
        gb_labels, gb_scores = [list(t) for t in zip(*top_gb_mean)]
        relu_labels, relu_scores = [list(t) for t in zip(*top_relu_mean)]
        sda_results_gb.append(gb_scores)
        sda_results_relu.append(relu_scores)
        x_vals = np.ones((n,),dtype=np.int) * (i + 1)
        ax.plot(x_vals.tolist(),sda_results_gb[i],'y*',label="GB SdA" if i == 0 else "_no_legend",markersize=9)
        ax.plot(x_vals.tolist(),sda_results_relu[i],'b*',label="ReLU SdA" if i == 0 else "_no_legend",markersize=9)
        
        #ax.errorbar(x,pca_means,yerr=pca_std, elinewidth=2, capsize=3, label="PCA", lw=1.5, fmt='--o')
        #ax.errorbar(x,lle_means,yerr=lle_std, elinewidth=2, capsize=3, label="LLE", lw=1.5, fmt='--o')
        #ax.errorbar(x,isomap_means,yerr=isomap_std, elinewidth=2, capsize=3, label="ISOMAP", lw=1.5, fmt='--o')
        #ax.errorbar(x,kpca_means,yerr=kpca_std, elinewidth=2, capsize=3, label="KPCA", lw=1.5, fmt='--o')        
        
    ax.xlim(0,len(dims_list) + 1)
    ax.ylim(0,0.50)
    ax.title(title)
    ax.xlabel('Dimension of the Data')
    ax.ylabel('Average Homogeneity')
    locs, labels = ax.xticks()   # get the xtick location and labels, re-order them so they match the experimental data
    labels = ['']
    labels.extend(dims_list)
    ax.xticks(locs,labels)
    ax.legend(loc = 7,numpoints=1)    # legend centre right    
           

        