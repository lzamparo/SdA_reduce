""" Utilities for data set extraction and scaling """
import tables
import os
import numpy as np
from optparse import OptionParser

from extract_datasets import draw_reference_population

# parse required args
parser = OptionParser()
parser.add_option("-i", "--input", dest="infile", help="read input h5 from here")
parser.add_option("-s", "--seed", dest="seed", type=int, help="use this random seed")
parser.add_option("-o", "--outfile", dest="outfile", help="specify the .h5 file that will contain the sampled data files")
(options, args) = parser.parse_args()

# set the random seed
np.random.seed(options.seed)

# open the infile
infile = tables.open_file(options.infile, 'r')

# open the outfile
group_name = "reference_seed_" + str(options.seed)
node_name = "reference_pop"
group_title = "The %d reference population sample" % (options.seed)
outfile = tables.open_file(options.outfile, 'a')
group = outfile.create_group('/', group_name, title=group_title)

# sample from the pop, with default params
reference_sample = draw_reference_population(infile)
zlib_filters = tables.Filters(complib='zlib', complevel=5)
atom = tables.Atom.from_dtype(reference_sample.dtype)

ref_h5 = outfile.create_carray(group, node_name, atom=atom, shape=reference_sample.shape, filters=zlib_filters)
ref_h5[:] = reference_sample
outfile.flush()

# close the input file
infile.close()

# write to outfile
outfile.close()