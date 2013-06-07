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

def pretrain(shared_args, private_args, num_epochs=50, batch_size=20): 
    """ Pretrain an SdA model for the given number of training epochs, changing one
    particular hyper-parameter as specified in private_args.

    :type shared_args: list
    :param shared_args: one item list whose first argument is a dictionary
    of the arguments shared by both child processes

    :type private_args: dict
    :param private_args: dictionary containing both the name and value
    of the argument in (momentum|corruption|weight_decay|learning_rate)
    to be changed from the value in shared_args for this child process
    
    :type num_epochs: int
    :param num_epochs: train for this many epochs
    
    :type batch_size: int
    :param batch_size: train in mini-batches of this size"""
    
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
    
    #output_filename = "stacked_denoising_autoencoder_" + private_args['arch'] + "." + day + "." + hour
    output_filename = "hyperparam_search." + private_args['param'] + "." + str(private_args['val']) + ".out"
    output_file = open(output_filename, 'w')
    os.chdir(current_dir)
    print >> output_file, "Run on " + str(datetime.now())  
    print '... building the model'
    
    # Set the particular argument (momentum|corruption|weight_decay|learning_rate)
    # based on the value in private_args
    shared_args_dict[private_args['param']] = private_args['val']
    
    corruption_list = [float(shared_args_dict['corruption']) for i in shared_args_dict['arch']]
    dA_losses = ['xent' for i in arch_list]
    dA_losses[0] = 'squared'
    sda = SdA(numpy_rng=numpy_rng, n_ins=n_features,
          hidden_layers_sizes=shared_args_dict['arch'],
          corruption_levels = corruption_list,
          dA_losses=dA_losses,              
          n_outs=3)    
    
    # Get the training data sample from the input file
    data_set_file = openFile(str(options.inputfile), mode = 'r')
    datafiles = extract_unlabeled_chunkrange(data_set_file, num_files = 10)
    train_set_x = load_data_unlabeled(datafiles)
    data_set_file.close()

    # compute number of minibatches for training, validation and testing
    n_train_batches, n_features = train_set_x.get_value(borrow=True).shape
    n_train_batches /= batch_size
    
    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'


    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    for i in xrange(sda.n_layers):
        
        for epoch in xrange(num_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_list[i],
                         lr=shared_args_dict["learning_rate"],
                         momentum=shared_args_dict["momentum"],
                         weight_decay=shared_args_dict["weight_decay"]))
            print >> output_file, 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print >> output_file, numpy.mean(c)

    end_time = time.clock()

    print >> output_file, ('Pretraining time for file ' +
                          os.path.split(__file__)[1] +
                          ' was %.2fm to go through %i epochs' % (((end_time - start_time) / 60.), (num_epochs / 2)))
  
    
    output_file.close()        
    
    

if __name__ == '__main__':
    
    # Parse command line args
    parser = OptionParser()
    parser.add_option("-d", "--dir", dest="dir", help="output directory")   
    parser.add_option("-i", "--inputfile", dest="inputfile", help="the data (hdf5 file) prepended with an absolute path")
    parser.add_option("-o", "--offset", dest="offset", type="int", help="use this offset for reading input from the hdf5 file")
    parser.add_option("-a", "--firstarg", dest="first", type='float', default=0., help="the first dynamic arg to set")
    parser.add_option("-b", "--secongarg", dest="second", type='float', default=0., help="the second dynamic arg to set")
    parser.add_option("-s", "--argtoset", dest="arg", help="specifies which parameter are we searching")
    (options, args) = parser.parse_args()    
    
    # Construct a dict of shared arguments that should be read by both processes
    manager = Manager()

    args = manager.list()
    args.append({})
    shared_args = args[0]
    shared_args['dir'] = options.dir
    shared_args['input'] = options.inputfile
    shared_args['corruption'] = 0.1
    shared_args['momentum'] = 0.
    shared_args['weight_decay'] = 0.0001
    shared_args['learning_rate'] = 0.01
    shared_args['arch'] = [800, 700, 300, 50]
    shared_args['offset'] = options.offset
    args[0] = shared_args
    
    # Construct the specific args for each of the two processes
    p_args = {}
    q_args = {}

    p_args['param'] = options.arg
    p_args['val'] = options.first
    q_args['param'] = options.arg
    q_args['val'] = options.second
        
    # Set the gpu for each sub-process
    p_args['gpu'] = 'gpu0'
    q_args['gpu'] = 'gpu1'

    # Run both sub-processes
    p = Process(target=pretrain, args=(args,p_args,))
    q = Process(target=pretrain, args=(args,q_args,))
    p.start()
    q.start()
    p.join()
    q.join()

    