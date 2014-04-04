from optparse import OptionParser
import os, re

import cPickle
import gzip
import sys
import time

import numpy
from scipy.linalg import norm

from extract_datasets import extract_unlabeled_chunkrange
from load_shared import load_data_unlabeled
from tables import openFile

from datetime import datetime

def test_gradient_SdA(private_args,finetune_lr=0.01, momentum=0.3, weight_decay = 0.0001, finetuning_epochs=5,
             batch_size=1000):
    # Import sandbox.cuda to bind the specified GPU to this subprocess
    # then import the remaining theano and model modules.
    import theano.sandbox.cuda
    theano.sandbox.cuda.use(private_args['gpu'])
    
    import theano
    import theano.tensor as T
    from theano.tensor.shared_randomstreams import RandomStreams
    
    from SdA import SdA    
        
    current_dir = os.getcwd()    
    os.chdir(private_args['dir'])
    today = datetime.today()
    day = str(today.date())
    hour = str(today.time())   
    output_filename = "finetune_sda_" + private_args['arch'] + "." + day + "." + hour
    output_file = open(output_filename,'w')
    os.chdir(current_dir)    
    print >> output_file, "Run on " + str(datetime.now())    
    
    # Get the training and validation data samples from the input file
    data_set_file = openFile(str(private_args['input']), mode = 'r')
    datafiles = extract_unlabeled_chunkrange(data_set_file, num_files = 25, offset = private_args['offset'])
    train_set_x = load_data_unlabeled(datafiles)
    validation_datafiles = extract_unlabeled_chunkrange(data_set_file, num_files = 5, offset = private_args['offset'] + 25)
    valid_set_x = load_data_unlabeled(validation_datafiles)    
    data_set_file.close()

    # compute number of minibatches for training, validation and testing
    n_train_batches, n_features = train_set_x.get_value(borrow=True).shape
    #DEBUG
    print >> output_file, "elements, features are: " + str(n_train_batches) + ", " + str(n_features)
    n_train_batches /= batch_size
    
    print >> output_file, "number of training batches: " + str(n_train_batches)
    
    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    
    print >> output_file, 'Unpickling the model from %s ...' % (private_args['restore'])        
    f = file(private_args['restore'], 'rb')
    sda = cPickle.load(f)
    f.close() 
    
    print >> output_file, "sucessfully unpickled the model."
    output_file.close()
    

def extract_arch(filename, model_regex):
    ''' Return the model architecture of this filename
    Modle filenames look like SdA_1000_500_100_50.pkl'''
    match = model_regex.match(filename)
    if match is not None:    
        return match.groups()[0]
    
if __name__ == '__main__':
        
    # Parse command line args
    parser = OptionParser()
    parser.add_option("-d", "--dir", dest="dir", help="base output directory")
    parser.add_option("-e", "--pretrain_experiment", dest="experiment", help="directory name containing pre-trained pkl files for this experiment (below the base directory)")
    parser.add_option("-x", "--output_extension", dest="extension", help="output directory name below the base, named for this finetuning experiment")
    parser.add_option("-p","--firstrestorefile",dest = "pr_file", help = "Restore the first model from this pickle file", default=None)
    parser.add_option("-i", "--inputfile", dest="inputfile", help="the data (hdf5 file) prepended with an absolute path")
    parser.add_option("-o", "--offset", dest="offset", type="int", help="use this offset for reading input from the hdf5 file")
    (options, args) = parser.parse_args()    
    
    # Construct the specific args dict
    p_args = {}
    
    p_args['dir'] = os.path.join(options.dir,options.extension)
    p_args['input'] = options.inputfile
    p_args['offset'] = options.offset
    p_args['gpu'] = 'gpu0'
    
    # Compile regular expression for extracting model architecture names
    model_name = re.compile(".*?_([\d_]+).pkl")    
    p_args['arch'] = extract_arch(options.pr_file,model_name)
   
         
    # Determine where to load & save the first model
    parts = os.path.split(options.dir)
    pkl_load_file = os.path.join(parts[0],'pretrain_pkl_files',options.experiment,options.pr_file)
    pkl_save_file = os.path.join(parts[0],'finetune_pkl_files',options.extension,options.pr_file)
    p_args['restore'] = pkl_load_file
    p_args['save'] = pkl_save_file

    # Try to unpickle the model
    test_gradient_SdA(p_args)