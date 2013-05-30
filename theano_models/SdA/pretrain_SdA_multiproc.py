""" SdA pretraining script that uses two GPUs, one per sub-process,
via the Python multiprocessing module.  """


# These imports will not trigger any theano GPU binding, so are safe to sit here.
from multiprocessing import Process, Manager
from optparse import OptionParser
import os

import cPickle
import gzip
import sys
import time

import numpy

from extract_datasets import extract_unlabeled_chunkrange
from load_shared import load_data_unlabeled
from tables import openFile

from datetime import datetime

def pretrain(shared_args,private_args,pretraining_epochs=50, pretrain_lr=0.001, batch_size=20): 
    """ Pretrain an SdA model for the given number of training epochs.  The model is either initialized from 
    scratch, or is reconstructed from a previously pickled model.

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type batch_size: int
    :param batch_size: train in mini-batches of this size """
    
    # Import sandbox.cuda to bind the specified GPU to this subprocess
    # then import the remaining theano and model modules.
    import theano.sandbox.cuda
    theano.sandbox.cuda.use(private_args['gpu'])
    
    import theano
    import theano.tensor as T
    from theano.tensor.shared_randomstreams import RandomStreams
    
    from mlp.logistic_sgd import LogisticRegression
    from mlp.hidden_layer import HiddenLayer
    from dA.AutoEncoder import AutoEncoder
    from SdA import SdA    
    
    shared_args_dict = shared_args[0]
    
    current_dir = os.getcwd()    
    
    os.chdir(shared_args_dict['dir'])
    today = datetime.today()
    day = str(today.date())
    hour = str(today.time())
    output_filename = "stacked_denoising_autoencoder_" + private_args['arch'] + "." + day + "." + hour
    output_file = open(output_filename,'w')
    os.chdir(current_dir)    
    print >> output_file, "Run on " + str(datetime.now())    
    
    # Get the training data sample from the input file
    data_set_file = openFile(str(shared_args_dict['input']), mode = 'r')
    datafiles = extract_unlabeled_chunkrange(data_set_file, num_files = 15, offset = shared_args_dict['offset'])
    train_set_x = load_data_unlabeled(datafiles)
    data_set_file.close()

    # compute number of minibatches for training, validation and testing
    n_train_batches, n_features = train_set_x.get_value(borrow=True).shape
    n_train_batches /= batch_size
    
    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    
    
    # Check if we can restore from a previously trained model,    
    # otherwise construct a new SdA
    if private_args.has_key('restore'):
        print >> output_file, 'Unpickling the model from %s ...' % (private_args['restore'])
        current_dir = os.getcwd()    
        os.chdir(shared_args_dict['dir'])         
        f = file(private_args['restore'], 'rb')
        sda = cPickle.load(f)
        f.close()        
        os.chdir(current_dir)
    else:
        print '... building the model'
        arch_list_str = private_args['arch'].split("-")
        arch_list = [int(item) for item in arch_list_str]
        corruption_list = [shared_args_dict['corruption'] for i in arch_list]
        dA_losses = ['xent' for i in arch_list]
        dA_losses[0] = 'squared'
        sda = SdA(numpy_rng=numpy_rng, n_ins=n_features,
              hidden_layers_sizes=arch_list,
              corruption_levels = corruption_list,
              dA_losses=dA_losses,              
              n_outs=3)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    print '... pre-training the model'
    start_time = time.clock()
    
    ## Pre-train layer-wise
    corruption_levels = sda.corruption_levels
    for i in xrange(sda.n_layers):
        
        # TODO: Set the learning rates to use.  See Yann Lecun's paper for backprop learning rate settings.
                
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print >> output_file, 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print >> output_file, numpy.mean(c)
            
        if private_args.has_key('save'):
            print >> output_file, 'Pickling the model...'
            current_dir = os.getcwd()    
            os.chdir(shared_args_dict['dir'])            
            f = file(private_args['save'], 'wb')
            cPickle.dump(sda, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()
            os.chdir(current_dir)

    end_time = time.clock()

    print >> output_file, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    output_file.close()        
    
    

if __name__ == '__main__':
    
    # Parse command line args
    parser = OptionParser()
    parser.add_option("-d", "--dir", dest="dir", help="output directory")   
    parser.add_option("-p","--firstrestorefile",dest = "pr_file", help = "Restore the first model from this pickle file", default=None)
    parser.add_option("-q","--secondrestorefile",dest = "qr_file", help = "Restore the second model from this pickle file", default=None)
    parser.add_option("-i", "--inputfile", dest="inputfile", help="the data (hdf5 file) prepended with an absolute path")
    parser.add_option("-c", "--corruption", dest="corruption", type="float", help="use this amount of corruption for the dA")
    parser.add_option("-o", "--offset", dest="offset", type="int", help="use this offset for reading input from the hdf5 file")
    parser.add_option("-a", "--firstarch", dest="p_arch", default = "", help="dash separated list to specify the first architecture of the SdA.  E.g: -a 850-400-50")
    parser.add_option("-b", "--secondarch", dest="q_arch", default = "", help="dash separated list to specify the second architecture of the SdA.")
    (options, args) = parser.parse_args()    
    
    # Construct a dict of shared arguments that should be read by both processes
    manager = Manager()

    args = manager.list()
    args.append({})
    shared_args = args[0]
    shared_args['dir'] = options.dir
    shared_args['input'] = options.inputfile
    shared_args['corruption'] = options.corruption
    shared_args['offset'] = options.offset
    args[0] = shared_args
    
    # Construct the specific args for each of the two processes
    p_args = {}
    q_args = {}

    # Determine where to save the model
    if options.pr_file is not None:
        # Save over the old restorefile
        p_args['save'] = options.pr_file
        p_args['restore'] = options.pr_file
    else:
        # Write to pkl file whose name derives from the specified architecture
        savename = options.p_arch
        p_args['save'] = "SdA_" + savename.replace('-','_') + ".pkl"
            
    if options.qr_file is not None:
        q_args['save'] = options.qr_file
        q_args['restore'] = options.qr_file
    else:
        savename = options.q_arch
        q_args['save'] = "SdA_" + savename.replace('-','_') + ".pkl"
        
    p_args['arch'] = options.p_arch
    p_args['gpu'] = 'gpu0'
    q_args['arch'] = options.q_arch
    q_args['gpu'] = 'gpu1'

    # Run both sub-processes
    p = Process(target=pretrain, args=(args,p_args,))
    q = Process(target=pretrain, args=(args,q_args,))
    p.start()
    q.start()
    p.join()
    q.join()

    