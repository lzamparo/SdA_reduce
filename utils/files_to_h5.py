#! /usr/bin/env python

# Vacuum up all the object.CSV files from the given input directory, and pack them into an hdf5 file (maybe using pytables later?)
# 
# 

from os import listdir, chdir, getcwd
from optparse import OptionParser

from tables.file import File, openFile, copyFile, hdf5Extension
from tables import Filters
from tables import Atom

from numpy import genfromtxt


# Check that options are present, else print help msg
parser = OptionParser()
parser.add_option("-i", "--input", dest="indir", help="read input from here")
parser.add_option("-s", "--suffix", dest="suffix", help="specify the suffix for data files")
parser.add_option("-f", "--filename", dest="filename", help="specify the .h5 filename that will contain all the data")
(options, args) = parser.parse_args()

# Open and prepare an hdf5 file 
filename = options.filename
h5file = openFile(filename, mode = "w", title = "Data File")

# Create a new group under "/" (root)
arrays_group = h5file.createGroup("/", 'recarrays', 'The object data arrays')
zlib_filters = Filters(complib='zlib', complevel=5)

# Go and read the files, 
input_dir = options.indir
suffix = options.suffix
cur_dir = getcwd()
try:
	files = listdir(input_dir)
	chdir(input_dir)
	
except:
	print "Could not read files from " + input_dir
else:
	for f in files:
		if f.endswith(suffix):
			data_range = f.split('.')[0]
			my_data = genfromtxt(f, delimiter=',', autostrip = True)
			atom = Atom.from_dtype(my_data.dtype)
			ds = h5file.createCArray(where=arrays_group, name=data_range, atom=atom, shape=my_data.shape, filters=zlib_filters)
			ds[:] = my_data
			h5file.flush()
	chdir(cur_dir)

h5file.close()


		

