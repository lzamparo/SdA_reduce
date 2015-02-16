#! /usr/bin/env python

""" 
Pass all cells in all wells of the input h5 file through the filters provided.  The output is another h5 file whose
tree mimics the input, but contains only the filtered cells.
"""

import os
from optparse import OptionParser

from load_shared import apply_constraints

from tables.file import File, open_file
from tables import Filters
from tables import Atom

import numpy as np

# Check that options are present, else print help msg
parser = OptionParser()
parser.add_option("-i", "--input", dest="infile", help="read input h5 from here")
parser.add_option("-f", "--filters", dest="filters", help="read the filters from here")
parser.add_option("-o", "--filename", dest="filename", help="specify the .h5 filename that will contain all the filtered data")
(options, args) = parser.parse_args()

# Open and prepare input and output hdf5 files 
filename = options.filename
h5output = open_file(filename, mode = "w", title = "Filtered Data File")
zlib_filters = Filters(complib='zlib', complevel=5)

h5input = open_file(options.infile, mode = "r")

# Create a new group under "/" (root)
plates_group = h5output.createGroup("/", 'plates', 'the plates for this replicate')

all_plates = [p._v_name for p in h5input.walk_groups("/plates")]
all_plates = all_plates[1:]

# Create a group for each plate in the output file
for plate in all_plates:
    desc = "plate number " + plate
    h5output.create_group("/plates/",plate,desc)

# Walk the input file, filtering each well we encounter, save to the output file
for p in all_plates:
    plate_group = "/plates/" + p
    print "processing plate %s " % (p)
    for w in h5input.walk_nodes(where=plate_group, classname='EArray'):
        well_name = w._v_name
        raw_data = w.read()
        filtered_data = apply_constraints(raw_data, constraints_file=options.filters)
        atom = Atom.from_dtype(filtered_data.dtype)
        if (filtered_data.shape[0] > 0):
            ds = h5output.create_carray(where=plate_group, name=well_name, atom=atom, shape=filtered_data.shape, filters=zlib_filters)
            ds[:] = filtered_data
        else:
            ds = h5output.create_earray(where=plate_group, name=well_name, atom=atom, shape=(0,filtered_data.shape[1]), filters=zlib_filters)
        h5output.flush()
print "done writing to h5 output file"
h5output.close()
h5input.close()



