#! /usr/env/python

import os, sys
import re
from numpy import log
import matplotlib.pyplot as plt

# cd into the tld
tld = '/data/dA_results'
#'sys.argv[1]
currdir = os.listdir('.')

# ignore the current directory ('.')
ignore = ['.']

# the RE for extracting corruption params from filenames
corruption_re = re.compile('^([\w]{2,4})\_[\w|\.]+\_([\d\.]{3,4})[\_|\.]')

# extract the final reconstruction error score and corruption values for the given file
def parse_file(filename, dirname):
    with open(os.path.join(dirname,filename), mode='r', buffering=1) as f:
        unit,corruption = strip_corruption(filename, corruption_re)
        lines = f.readlines()
        line = lines[-2]
        parts = line.split()
        score = parts[-1]
    return unit, float(score), float(corruption)


# parse the filenames for recovering corruption parameters
def strip_corruption(filename,regex):
    match = regex.match(filename)
    if match is not None:
        if len(match.groups()) < 3:
            return match.groups()[0],match.groups()[1]
        else:
            return None 

# set up initial results dict
results = {'gb': [], 'relu': []}

# for dirname, dirlist, filenames in os.walk('.'):
print "....parsing the results: "
for dirpath, dirnames, filenames in os.walk(tld):
    if dirpath == '.':
        continue
    for infile in filenames:
        unit, score, corruption = parse_file(infile,dirpath)
        results[unit].append((corruption, score))

# plot the tuples
print "....plotting the results: "
colours = ['r--','b--']
plot_objs = [None,None]
labels = [0,0]
for i,unit in enumerate(results.keys()):
    ordered_data = sorted(results[unit], key=lambda t: t[0])
    x_vals = [item[0] for item in ordered_data]
    y_vals = [log(item[1]) for item in ordered_data]
    labels[i] = unit
    plt.plot(x_vals,y_vals,colours[i])
               
plt.xlabel('Corruption')
plt.ylabel('Reconstruction Error (log scale)')
plt.title('ReLU unit vs Gaussian-Bernoulli unit dAs')    
plt.legend(tuple(labels), 'lower right', shadow=True, fancybox=True)
plt.savefig("relu_vs_gb.pdf", dpi=100, format="pdf")
