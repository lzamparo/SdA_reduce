""" SdA compression script.  Take data input from an hdf5 file, output the highest
layer dA output (i.e lowest dimensional representation) of that data to an output hdf5 file. """


# These imports will not trigger any theano GPU binding, so are safe to sit here.
from multiprocessing import Process, Manager
from optparse import OptionParser
import os, re

import cPickle
import gzip
import sys
import time

import numpy
    
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from SdA import SdA

from extract_datasets import extract_unlabeled_chunkrange, store_unlabeled_byarray
from load_shared import load_data_unlabeled
from tables import openFile, Filters



from datetime import datetime

def feedforward_SdA(output_dir,input_file,arch,restore_file): 
    """ Feed the data through the SdA 
    
    :type shared_args: list
    :param shared_args: list contaning a dict of shared arguments 
    provided from the parent process

    :type private_args: dict
    :param private_args: dict containing the arguments unique to 
    this child process
    
    """
    
    # Open and set up the output hdf5 file
    current_dir = os.getcwd()    
    os.chdir(output_dir)
    today = datetime.today()
    day = str(today.date())
    hour = str(today.time())   
    output_filename = "reduce_SdA." + arch + "." + day + "." + hour + ".h5"
    h5file = openFile(output_filename, mode = "w", title = "Data File")    
    os.chdir(current_dir)  
    
    print "Run on " + str(datetime.now())    
    
    # Create a new group under "/" (root)
    arrays_group = h5file.createGroup("/", 'recarrays', 'The lower dimensional data arrays')
    zlib_filters = Filters(complib='zlib', complevel=5)    
    
    # Get the data to be fed through the SdA from the input file
    data_set_file = openFile(input_file, mode = 'r') 
    arrays_list = data_set_file.listNodes("/recarrays", classname='Array')
    chunk_names, offsets = calculate_offsets(arrays_list)
    
    print 'Unpickling the model from %s ...' % (restore_file)        
    f = file(restore_file, 'rb')
    sda = cPickle.load(f)
    f.close()    
    
    datafile = extract_unlabeled_chunkrange(data_set_file, num_files=len(arrays_list))
    this_x = load_data_unlabeled(datafile)    
    
    print '... getting the encoding function'
    encode_fn = sda.build_encoding_functions(dataset=this_x)    
    
    start_time = time.clock()
    # Go through each chunk in the data_set_file, feed through the SdA, write the output to h5file
    for i in xrange(len(arrays_list)):
        start,end = offsets[i]
        reduced_data = encode_fn(this_x,start=start,end=end)
        store_unlabeled_byarray(h5file, arrays_group, zlib_filters, chunk_names[i], reduced_data.get_value())
        
    # tidy up    
    end_time = time.clock()
    data_set_file.close()
    h5file.close()       
    

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
    parser.add_option("-d", "--dir", dest="dir", help="base output directory")
    parser.add_option("-x", "--output_extension", dest="extension", help="output directory name below the base, named for this finetuning experiment")
    parser.add_option("-r","--restorefile",dest = "pr_file", help = "Restore the first model from this pickle file", default=None)
    parser.add_option("-i", "--inputfile", dest="inputfile", help="the data (hdf5 file) prepended with an absolute path")
    (options, args) = parser.parse_args()    
    
    parts = os.path.split(options.dir)
    output_dir = os.path.join(options.dir,options.extension)
    input_file = options.inputfile
    
    model_name = re.compile(".*?_([\d_]+).pkl")    
    arch = extract_arch(options.pr_file,model_name)
    
    restore_file = os.path.join(parts[0],'finetune_pkl_files',options.extension,options.pr_file)
    
    feedforward_SdA(output_dir, input_file, arch, restore_file)

    