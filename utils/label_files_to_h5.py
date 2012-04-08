#!/usr/bin/env python

"""
The data set that is in the specified hdf5 file has labels in two different locations.  
The label input files have the .label suffix.
The foci input files have the .out suffix.

Vacuum up the labels from the two given directories, merge the relevant portions (image number,object number) and
convert from two-dimensional binary labeling into one multi class label, by the following scheme:

-1,-1 -> 0
1,-1 -> 1
-1,1 -> 2
1,1 -> 3
"""
from os import listdir, chdir, getcwd
from optparse import OptionParser

from tables.file import File, openFile, copyFile, hdf5Extension
from tables import Filters
from tables import Atom

import numpy as np

# Check that options are present, else print help msg.
parser = OptionParser()
parser.add_option("--labelinput", dest="labels_indir", help="read shape labels from here")
parser.add_option("--fociinput", dest="foci_indir", help="read foci labels from here")
parser.add_option("-f", "--filename", dest="filename", help="specify the .h5 filename that will contain the data labels")
(options, args) = parser.parse_args()

# label conversion dictionary
label_dict = {(-1.0,-1.0): 0, (1.0,-1.0): 1, (-1.0,1.0): 2, (1.0,1.0): 3}

''' Read files in directory loc, return a list of those ending with suffix '''
def read_filenames(loc,suffix):
    files = listdir(loc)
    selected = []
    for f in files:
        if f.endswith(suffix):
            selected.append(f)
    return selected

''' Read the given file (in directory dir) and return the numpy array within '''
def read_array_from_file(filename,dirname):
    chdir(dirname)
    array = np.genfromtxt(filename, delimiter=',', autostrip=True, usecols=(0,1,2))
    return array
    
''' Take two arrays, merge them together after translating the labels which are in the first 
    column of each array
'''
def combine_arrays(foci_array,shape_array):
    foci_labels = foci_array[:,0]
    shape_labels = shape_array[:,0]
    new_labels = np.array([label_dict[t] for t in zip(foci_labels,shape_labels)])
    new_labels.shape = (len(new_labels),1)
    return np.hstack((new_labels,foci_array[:,(1,2)]))
    
    
        

# Open and prepare an hdf5 file, adding a labels group 
filename = options.filename
h5file = openFile(filename, mode = "a", title = "Data File")
labels_group = h5file.createGroup("/", 'labels', 'The labels and object IDs')
zlib_filters = Filters(complib='zlib', complevel=5)

# Go to the files location in the filesystem.
shape_input = options.labels_indir
foci_input = options.foci_indir
cur_dir = getcwd()

try:
    foci_files = read_filenames(foci_input,'.out')
    shape_files = read_filenames(shape_input,'.label')
except:
    print "Could not read files from one of " + foci_input + ", " + shape_input
    sys.exit(1)

foci_files.sort()
shape_files.sort()

# iterate over sorted & zipped files
for (f,s) in zip(foci_files,shape_files):
    label_range = f.split('.')[0]
    foci_array = read_array_from_file(f,foci_input)
    shape_array = read_array_from_file(s,shape_input)
    relabeled_array = combine_arrays(foci_array,shape_array)
    atom = Atom.from_dtype(relabeled_array.dtype)
    labels = h5file.createCArray(where=labels_group, name=label_range, atom=atom, shape=relabeled_array.shape, filters=zlib_filters)
    labels[:] = relabeled_array
    h5file.flush()


# Close the h5 file when done.
h5file.close()
