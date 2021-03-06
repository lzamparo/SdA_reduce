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

from extract_datasets import extract_unlabeled_chunkrange, extract_labeled_chunkrange
from extract_datasets import store_unlabeled_byarray, store_labeled_byarray
from common_utils import extract_arch
from load_shared import load_data_unlabeled
from tables import openFile, Filters 

from datetime import datetime

def feedforward_SdA(shared_args,private_args): 
    """ Feed the data through the SdA 
    
    :type shared_args: list
    :param shared_args: list contaning a dict of shared arguments 
    provided from the parent process

    :type private_args: dict
    :param private_args: dict containing the arguments unique to 
    this child process
    
    """
    
    # Import sandbox.cuda to bind the specified GPU to this subprocess
    # then import the remaining theano and model modules.
    import theano.sandbox.cuda
    theano.sandbox.cuda.use(private_args['gpu'])
    
    import theano
    import theano.tensor as T
    from theano.tensor.shared_randomstreams import RandomStreams
    
    from SdA import SdA    
    
    shared_args_dict = shared_args[0]
    
    # Open and set up the output hdf5 file
    current_dir = os.getcwd()    
    os.chdir(shared_args_dict['dir'])
    today = datetime.today()
    day = str(today.date())
    hour = str(today.time())   
    output_filename = "reduce_SdA." + private_args['arch'] + "." + day + "." + hour + ".h5"
    h5file = openFile(output_filename, mode = "w", title = "Data File")    
    os.chdir(current_dir)  
    
    print "Run on " + str(datetime.now())    
    
    # Create a new group under "/" (root)
    save_labels = shared_args_dict['labels']
    arrays_group = h5file.createGroup("/", 'recarrays', 'The lower dimensional data arrays')
    if save_labels:
        labels_group = h5file.createGroup("/", 'labels', 'The label arrays')
    zlib_filters = Filters(complib='zlib', complevel=5)    
    
    # Get the data to be fed through the SdA from the input file
    data_set_file = openFile(str(shared_args_dict['input']), mode = 'r') 
    arrays_list = data_set_file.listNodes("/recarrays", classname='Array')
    if save_labels:
        labels_list = data_set_file.listNodes("/labels", classname='Array')
    chunk_names, offsets = calculate_offsets(arrays_list)
    
    print 'Unpickling the model from %s ...' % (private_args['restore'])        
    f = file(private_args['restore'], 'rb')
    sda_model = cPickle.load(f)
    f.close()    
    
    if save_labels:
        datafile, labelfile = extract_labeled_chunkrange(data_set_file, num_files=len(arrays_list))
    else:
        datafile = extract_unlabeled_chunkrange(data_set_file, num_files=len(arrays_list))
    this_x = load_data_unlabeled(datafile)  
    
    print '... getting the encoding function'
    encode_fn = sda_model.build_encoding_functions(dataset=this_x)    
    
    start_time = time.clock()
    # Go through each chunk in the data_set_file, feed through the SdA, write the output to h5file
    for i in xrange(len(arrays_list)):
        start,end = offsets[i]
        reduced_data = encode_fn(start=start,end=end)
        if save_labels:
            store_labeled_byarray(h5file, arrays_group, labels_group, zlib_filters, chunk_names[i], reduced_data, labels_list[i])
        else:
            store_unlabeled_byarray(h5file, arrays_group, zlib_filters, chunk_names[i], reduced_data)
        
    # tidy up    
    end_time = time.clock()
    data_set_file.close()
    h5file.close()       
    

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
    parser.add_option("-p","--firstrestorefile",dest = "pr_file", help = "Restore the first model from this pickle file", default=None)
    parser.add_option("-q","--secondrestorefile",dest = "qr_file", help = "Restore the second model from this pickle file", default=None)
    parser.add_option("-i", "--inputfile", dest="inputfile", help="the data (hdf5 file) prepended with an absolute path")
    parser.add_option("-l", "--labels", dest="labels", action='store_true', default=False, help="use labels?")
    (options, args) = parser.parse_args()    
    
    # Construct a dict of shared arguments that should be read by both processes
    manager = Manager()

    args = manager.list()
    args.append({})
    shared_args = args[0]
    shared_args['dir'] = os.path.join(options.dir,options.extension)
    shared_args['input'] = options.inputfile
    shared_args['labels'] = options.labels
    args[0] = shared_args
    
    # Construct the specific args for each of the two processes
    p_args = {}
    q_args = {}
       
    p_args['gpu'] = 'gpu0'
    q_args['gpu'] = 'gpu1'
    
    # Compile regular expression for extracting model architecture names
    model_name = re.compile(".*?_([\d_]+).pkl")    
    p_args['arch'] = extract_arch(options.pr_file,model_name)
    q_args['arch'] = extract_arch(options.qr_file,model_name)
         
    # Determine where to load & save the first model
    parts = os.path.split(options.dir)
    pkl_load_file = os.path.join(parts[0],'finetune_pkl_files',options.extension,options.pr_file)
    p_args['restore'] = pkl_load_file
   
    # Determine where to load & save the second model
    pkl_load_file = os.path.join(parts[0],'finetune_pkl_files',options.extension,options.qr_file)
    q_args['restore'] = pkl_load_file

    # Run both sub-processes
    p = Process(target=feedforward_SdA, args=(args,p_args,))
    q = Process(target=feedforward_SdA, args=(args,q_args,))
    p.start()
    q.start()
    p.join()
    q.join()

    