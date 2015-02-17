""" SdA compression script.  Take data input from an hdf5 file, output the highest
layer dA output (i.e lowest dimensional representation) of that data to an output hdf5 file. """


# These imports will not trigger any theano GPU binding, so are safe to sit here.
from optparse import OptionParser
import os, re

import cPickle
import gzip
import time

import numpy
import tables
    
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from SdA import SdA

from extract_datasets import store_unlabeled_byarray
from load_shared import load_data_unlabeled

from datetime import datetime

def feedforward_SdA(output_file,input_file,arch,restore_file): 
    """ Walk through the input_file, feed each array through the SdA,
    save to a similar group/node structure in the output file.
    
    :type output_file: string
    :param output_file: write the h5 output here 

    :type input_file: string
    :param input_file: read the h5 input from here
    
    :type arch: string
    :param arch: string representing the architecture of the SdA used to reduce the data
    
    :type restore_file: string
    :param restore_file: location on disk of the pickled SdA model
    
    """
    
    # Open and set up the input, output hdf5 files     
    outfile_h5 = tables.openFile(output_file, mode = 'w', title = "Reduced Data File")    
    root = outfile_h5.createGroup('/','reduced_samples','reduced data from reference samples')
    input_h5 = tables.openFile(input_file, mode = 'r') 
    print "Run on ", str(datetime.now())    
    print "Reduced with ", arch
    
    # Create a new group under "/" (root)
    zlib_filters = tables.Filters(complib='zlib', complevel=5)   

    print 'Unpickling the model from %s ...' % (restore_file)        
    f = file(restore_file, 'rb')
    sda_model = cPickle.load(f)
    f.close()    
     
    # walk the node structure of the input, reduce, save to output
    for node in input_h5.walkNodes('/',classname='Array'):
        name = node._v_name
        try:
            data = node.read()
        except:
            print "Encountered a problem at this node: ", name
            continue
        # load the node data into theano.shared memory on the GPU
        this_x = load_data_unlabeled(data)   
        # get the encoding function, encode the data
        encode_fn = sda_model.build_encoding_functions(dataset=this_x)   
        start, end = 0, data.shape[0]
        reduced_data = encode_fn(start=start,end=end)     
        # write the encoded data back to the file
        store_unlabeled_byarray(outfile_h5, root, zlib_filters, name, reduced_data)
               
    h5file.close()
    outfile_h5.close()
           
def extract_arch(filename, model_regex):
    ''' Return the model architecture of this filename
    Modle filenames look like SdA_1000_500_100_50.pkl'''
    match = model_regex.match(filename)
    if match is not None:    
        return match.groups()[0]

def calculate_offsets(arrays_list):
    ''' Return the names and endpoint tuples of each chunk 
    in the arrays list '''
    chunk_names = [item.name for item in arrays_list]
    chunk_sizes = numpy.asarray([int(item.shape[0]) for item in arrays_list])
    offset_dict = {}
    
    # calculate the offsets for each data chunk
    end_pts = numpy.add.accumulate(chunk_sizes)
    start_pts = numpy.subtract(end_pts,chunk_sizes)
    for i in xrange(len(end_pts)):
        offset_dict[i] = (start_pts[i],end_pts[i])
        
    return chunk_names, offset_dict
    
    
if __name__ == '__main__':
        
    # Parse command line args
    parser = OptionParser()
    parser.add_option("-d","--sda_dir",dest="model_dir",help="directory containing the SdA model file(s)")
    parser.add_option("-r","--restorefile",dest = "pr_file", help = "Restore the first model from this pickle file", default=None)
    parser.add_option("-i", "--inputfile", dest="inputfile", help="the data (hdf5 file) prepended with an absolute path")
    parser.add_option("-o", "--outputfile", dest="outputfile", help="the output hdf5 file")
    (options, args) = parser.parse_args()    
    
    model_name = re.compile(".*?_([\d_]+).pkl")    
    arch = extract_arch(options.pr_file,model_name)
    
    restore_file = os.path.join(options.model_dir,options.pr_file)
    
    feedforward_SdA(options.outputfile, options.inputfile, arch, restore_file)

    