#! /usr/bin/env python

""" 
Vacuum up all the object.CSV files from the given input directory, and pack them into an hdf5 file that is organized by plate.well
Plates go 1 .. 14.  Rows go 1 ... 16, Cols 1 ... 24.
"""

import os
from optparse import OptionParser
import pandas

from tables.file import File, openFile
from tables import Filters
from tables import Atom

import numpy as np

# Check that options are present, else print help msg
parser = OptionParser()
parser.add_option("-i", "--input", dest="indir", help="read input from here")
parser.add_option("-s", "--suffix", dest="suffix", help="specify the suffix for data files")
parser.add_option("-d", "--dataframe", dest="dataframe", help="read a csv file describing the data set here")
parser.add_option("-o", "--filename", dest="filename", help="specify the .h5 filename that will contain all the data")
(options, args) = parser.parse_args()

# Open and prepare an hdf5 file 
filename = options.filename
h5file = openFile(filename, mode = "w", title = "Data File")

# Load the dataframe describing the layout of the experimental data
df = pandas.read_csv(options.dataframe)
all_plates = set(df['Plate'])

# Create a new group under "/" (root)
plates_group = h5file.createGroup("/", 'plates', 'the plates for this replicate')

# Create a group for each plate
for plate in all_plates:
    desc = "plate number " + str(plate)
    h5file.createGroup("/plates/",str(plate),desc)

# build a lookup of image number to plate, well
img_to_pw = {}

# populate the lookup table of image number to (plate, well)
for index, rec in df.iterrows():
    for img_num in xrange(rec['Low'],rec['High'] + 1):
        well = (int(rec['Row']) - 1) * 24 + int(rec['Col'])
        img_to_pw[img_num] = (rec['Plate'],well)

# get the root
root = h5file.root

# Go and read the files, 
input_dir = options.indir
suffix = options.suffix
cur_dir = os.getcwd()
try:
    files = os.listdir(input_dir)
    os.chdir(input_dir)
except:
    print "Could not read files from " + input_dir

# Read all the files, process 'em.
zlib_filters = Filters(complib='zlib', complevel=5)
for i,f in enumerate(files):
    if i % 10 == 0:
        print "processing %s, %d files done of %d total" % (f,i,len(files))
    if f.endswith(suffix):
        my_data = np.genfromtxt(f, delimiter=',', autostrip = True)
        atom = Atom.from_dtype(my_data.dtype)
        # slice this data file by grouped image numbers
        min_img, max_img = int(min(my_data[:,0])), int(max(my_data[:,0]))
        for img_num in xrange(min_img,max_img+1):
            try:
                plate, well = img_to_pw[img_num]
            except KeyError as e:
                print "image number not found in image to well map: " + img_num
                continue
            objs = my_data[my_data[:,0] == img_num]
            well_group = "/plates/" + str(plate)
            well_node = "/plates/" + str(plate) + "/" + str(well)
            if h5file.__contains__(well_node):
                # some data for this well exists in an EArray already, append this data to it.
                ds = h5file.get_node(where=well_node)
                ds.append(objs)				
            else:
                # no data from images belonging to this well have yet been dumped into an EArray.
                ds = h5file.create_earray(where=well_group, name=str(well), atom=atom, shape=(0,my_data.shape[1]), filters=zlib_filters)
                ds.append(objs)				
            h5file.flush()
os.chdir(cur_dir)
print "done!"
h5file.close()



