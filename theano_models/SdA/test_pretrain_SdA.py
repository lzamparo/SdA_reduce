import numpy
import cPickle

import theano
import theano.tensor as T
from mlp.logistic_sgd import LogisticRegression
from mlp.hidden_layer import HiddenLayer
from dA.AutoEncoder import AutoEncoder
from SdA import SdA

from theano.tensor.shared_randomstreams import RandomStreams

from extract_datasets import extract_unlabeled_chunkrange
from load_shared import load_data_unlabeled
from tables import openFile

import os
import sys
import time
from datetime import datetime
from optparse import OptionParser

def test_pickled_SdA(num_epochs=10, pretrain_lr=0.001, batch_size=10):
    """
    
    Pretrain an SdA model for the given number of training epochs.  The model is either initialized from scratch, or 
    is reconstructed from a previously pickled model.

    :type num_epochs: int
    :param num_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type batch_size: int
    :param batch_size: train in mini-batches of this size

    """
    
    current_dir = os.getcwd()    

    os.chdir(options.dir)
    today = datetime.today()
    day = str(today.date())
    hour = str(today.time())
    output_filename = "test_pickled_stacked_denoising_autoencoder_pretrain." + day + "." + hour
    output_file = open(output_filename,'w')
    os.chdir(current_dir)    
    print >> output_file, "Run on " + str(datetime.now())    
    
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

    sda = SdA(numpy_rng=numpy_rng, n_ins=n_features,
              hidden_layers_sizes=[850, 400, 50],
              corruption_levels = [0.1,0.1,0.1],
              n_outs=3, dA_losses=['squared','xent','xent'])

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    print 'Dumping pretraining functions to output file pre pickling...'
    print >> output_file, 'Pretraining functions, pre pickling'
    for i in xrange(sda.n_layers):
        theano.printing.debugprint(pretraining_fns[i], file = output_file, print_type=True) 

    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    corruption_levels = [float(options.corruption), float(options.corruption), float(options.corruption)]
    for i in xrange(sda.n_layers):
        
        for epoch in xrange(num_epochs / 2):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print >> output_file, 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print >> output_file, numpy.mean(c)

    end_time = time.clock()

    print >> output_file, ('Pretraining time for file ' +
                          os.path.split(__file__)[1] +
                          ' was %.2fm to go through %i epochs' % (((end_time - start_time) / 60.), (num_epochs / 2)))

    # Pickle the SdA
    print >> output_file, 'Pickling the model...'
    f = file(options.savefile, 'wb')
    cPickle.dump(sda, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()    
    
    # Unpickle the SdA
    print >> output_file, 'Unpickling the model...'
    f = file(options.savefile, 'rb')
    pickled_sda = cPickle.load(f)
    f.close()    

    # Regenerate the list of pretraining functions for the pickled SdA
    pretraining_fns = pickled_sda.pretraining_functions(train_set_x=train_set_x,
                                                    batch_size=batch_size)
    
    print 'Dumping pretraining functions to output file post unpickling...'
    print >> output_file, 'Pretraining functions, post unpickling'
    for i in xrange(sda.n_layers):
        theano.printing.debugprint(pretraining_fns[i], file = output_file, print_type=True)
    
    print >> output_file, 'Resume training...'
    start_time = time.clock()
    ## Pre-train layer-wise
    corruption_levels = [float(options.corruption), float(options.corruption), float(options.corruption)]
    for i in xrange(pickled_sda.n_layers):
        # pickle the current model, along with the current epoch run
        # TODO: pickling code here.
        
        for epoch in xrange(num_epochs / 2):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print >> output_file, 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print >> output_file, numpy.mean(c)

    end_time = time.clock()    
    print >> output_file, ('Pretraining time for file ' +
                          os.path.split(__file__)[1] +
                          ' was %.2fm to go through the remaining %i epochs' % (((end_time - start_time) / 60.), (num_epochs / 2)))
    
    # Test that the W-matrices and biases for the dA layers are all close to the W-matrices 
    # and biases for the MLP layers
    for i in xrange(pickled_sda.n_layers):
        dA_params = pickled_sda.dA_layers[i].get_params()
        MLP_params = pickled_sda.sigmoid_layers[i].get_params()
    
        if not numpy.allclose(dA_params[0].get_value(), MLP_params[0].get_value()):
            print >> output_file, ("numpy says that Ws in layer %i are not close" % (i))
        if not numpy.allclose(dA_params[1].get_value(), MLP_params[1].get_value()):
            print >> output_file, ("numpy says that the biases in layer %i are not close" % (i))  
    
    output_file.close()   
    
def test_unpickled_SdA(num_epochs=10, pretrain_lr=0.001, batch_size=10):
    """ Unpickle an SdA from file, continue pre-training for a given number of epochs.
    
    :type num_epochs: int
    :param num_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type batch_size: int
    :param batch_size: train in mini-batches of this size  
    """
    
    current_dir = os.getcwd()    
    
    os.chdir(options.dir)
    today = datetime.today()
    day = str(today.date())
    hour = str(today.time())
    output_filename = "test_unpickled_stacked_denoising_autoencoder_pretrain." + day + "." + hour
    output_file = open(output_filename,'w')
    os.chdir(current_dir)    
    print >> output_file, "Run on " + str(datetime.now())    
    
    # Get the training data sample from the input file
    data_set_file = openFile(str(options.inputfile), mode = 'r')
    datafiles = extract_unlabeled_chunkrange(data_set_file, num_files = 10)
    train_set_x = load_data_unlabeled(datafiles)
    data_set_file.close()

    # compute number of minibatches for training, validation and testing
    n_train_batches, n_features = train_set_x.get_value(borrow=True).shape
    n_train_batches /= batch_size    
    
    # Unpickle the SdA
    print >> output_file, 'Unpickling the model...'
    f = file(options.savefile, 'rb')
    pickled_sda = cPickle.load(f)
    f.close()    
    
    
    # Train for the remaining 
    pretraining_fns = pickled_sda.pretraining_functions(train_set_x=train_set_x,
                                                    batch_size=batch_size)
    
    print >> output_file, 'Resume training...'
    start_time = time.clock()
    ## Pre-train layer-wise
    corruption_levels = [float(options.corruption), float(options.corruption), float(options.corruption)]
    for i in xrange(pickled_sda.n_layers):
        # pickle the current model, along with the current epoch run
        # TODO: pickling code here.
        
        for epoch in xrange(num_epochs / 2):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print >> output_file, 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print >> output_file, numpy.mean(c)

    end_time = time.clock()    
    print >> output_file, ('Pretraining time for file ' +
                          os.path.split(__file__)[1] +
                          ' was %.2fm to go through the remaining %i epochs' % (((end_time - start_time) / 60.), (num_epochs / 2)))    

    
    # Test that the W-matrices for the dA layers are all close to the W-matrices for the MLP layers
    for i in xrange(pickled_sda.n_layers):
        dA_params = pickled_sda.dA_layers[i].get_params()
        MLP_params = pickled_sda.sigmoid_layers[i].get_params()
    
        if not numpy.allclose(dA_params[0].get_value(), MLP_params[0].get_value()):
            print >> output_file, ("numpy says that Ws in layer %i are not close" % (i))
        if not numpy.allclose(dA_params[1].get_value(), MLP_params[1].get_value()):
            print >> output_file, ("numpy says that the biases in layer %i are not close" % (i))  
    
    output_file.close()    
    
    
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--dir", dest="dir", help="test output directory")
    parser.add_option("-s","--savefile",dest = "savefile", help = "Save the model to this pickle file")
    parser.add_option("-r","--restorefile",dest = "restorefile", help = "Restore the model from this pickle file")
    parser.add_option("-i", "--inputfile", dest="inputfile", help="the data (hdf5 file) prepended with an absolute path")
    parser.add_option("-c", "--corruption", dest="corruption", help="use this amount of corruption for the dA s")
    
    (options, args) = parser.parse_args()        
    
    test_pickled_SdA()
    test_unpickled_SdA()
    
    