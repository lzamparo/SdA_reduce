""" SdA compression script.  Take data input from an hdf5 file, output the highest
layer dA output (i.e lowest dimensional representation) of that data to an output hdf5 file. """


# These imports will not trigger any theano GPU binding, so are safe to sit here.
from multiprocessing import Process, Manager
from optparse import OptionParser
import os, re

import cPickle
import gzip
import time

import numpy
import tables

from extract_datasets import store_unlabeled_byarray
from load_shared import load_data_unlabeled

from datetime import datetime

def feedforward_SdA(shared_args,private_args): 
    """ Walk through the input_file, feed each array through the SdA,
    save to a similar group/node structure in the output file.
    
    Feed the data through the SdA 
    
    :type shared_args: list
    :param shared_args: list contaning a dict of shared arguments 
    provided from the parent process

    :type private_args: dict
    :param private_args: dict containing the arguments unique to 
    this child process
    
    """
    
    # Import sandbox.cuda to bind the specified GPU to this subprocess
    # then import the remaining theano and model modules
    
    shared_args_dict = shared_args[0]   
    
    import theano.sandbox.cuda
    theano.sandbox.cuda.use(private_args['gpu'])
    
    import theano
    import theano.tensor as T
    from theano.tensor.shared_randomstreams import RandomStreams
    from SdA import SdA    
     
    # Open and set up the input, output hdf5 files     
    outfile_h5 = tables.openFile(private_args['output'], mode = 'w', title = "Reduced Data File")    
    root = outfile_h5.createGroup('/','reduced_samples','reduced data from reference samples')
    input_h5 = tables.openFile(str(shared_args_dict['input']), mode = 'r') 
    print "Run on ", str(datetime.now())    
    print "Reduced with ", private_args['arch']
    
    # Create a new group under "/" (root)
    zlib_filters = tables.Filters(complib='zlib', complevel=5)   

    print 'Unpickling the model from %s ...' % (private_args['restore'])        
    f = file(private_args['restore'], 'rb')
    sda_model = cPickle.load(f)
    f.close()    
    
    out_root = outfile_h5.root 
    # walk the node structure of the input, reduce, save to output
    for in_plate in input_h5.listNodes('/plates',classname="Group"):
        out_plate_name = in_plate._v_name
        out_plate_desc = in_plate._v_title
        # create this plate group in the output file
        out_plate = outfile_h5.createGroup(out_plates,out_plate_name,out_plate_desc)
        wells = in_plate._f_list_nodes(classname='Array')
        for well in wells:
            name = well._v_name
            parent_name = (well._v_parent)._v_name
            try:
                data = well.read()
                if data.shape[0] > 0:
                    # load the node data into theano.shared memory on the GPU
                    this_x = load_data_unlabeled(data)   
                    # get the encoding function, encode the data
                    encode_fn = sda_model.build_encoding_functions(dataset=this_x)   
                    start, end = 0, data.shape[0]
                    reduced_data = encode_fn(start=start,end=end)     
                else:
                    reduced_data = data[:,:10]
                store_unlabeled_byarray(outfile_h5, data_group, zlib_filters, name, reduced_data)                
                
            except:
                print "Encountered a problem at this node: ", name
                continue            
            # write reduced data to same place in outfile
            store_unlabeled_byarray(outfile_h5, out_plate, zlib_filters, name, reduced_data)                
               
    input_h5.close()
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
    parser.add_option("--p_dir",dest="p_model_dir",help="directory containing the first SdA model file")
    parser.add_option("--q_dir",dest="q_model_dir",help="directory containing the second SdA model file")
    parser.add_option("-p","--restorep",dest = "pr_file", help = "Restore the first model from this pickle file", default=None)
    parser.add_option("-q","--restoreq",dest = "qr_file", help = "Restore the second model from this pickle file", default=None)
    parser.add_option("-i", "--inputfile", dest="inputfile", help="the data (hdf5 file) prepended with an absolute path")
    parser.add_option("--p_out", dest="p_outputfile", help="the first model's output hdf5 file")
    parser.add_option("--q_out", dest="q_outputfile", help="the second model's output hdf5 file")
    (options, args) = parser.parse_args()    
    
    # Construct a dict of shared arguments that should be read by both processes
    manager = Manager()

    args = manager.list()
    args.append({})
    shared_args = args[0]
    shared_args['input'] = options.inputfile
    args[0] = shared_args
    
    # Construct the specific args for each of the two processes
    p_args = {}
    q_args = {}
       
    model_name = re.compile(".*?_([\d_]+).pkl")    
            
    p_args['gpu'] = 'gpu0'
    q_args['gpu'] = 'gpu1'
    
    p_args['arch'] = extract_arch(options.pr_file,model_name)
    q_args['arch'] = extract_arch(options.qr_file,model_name)
        
    p_args['restore'] = os.path.join(options.p_model_dir,options.pr_file)
    q_args['restore'] = os.path.join(options.q_model_dir,options.qr_file)

    p_args['output'] = options.p_outputfile
    q_args['output'] = options.q_outputfile

    # Run both sub-processes
    p = Process(target=feedforward_SdA, args=(args,p_args,))
    q = Process(target=feedforward_SdA, args=(args,q_args,))
    p.start()
    q.start()
    p.join()
    q.join()            
        
    
    
    
    

    