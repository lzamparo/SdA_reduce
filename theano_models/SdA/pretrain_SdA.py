"""
 This script pre-trains a stacked de-noising auto-encoder.  
 The SdA model is either recovered from a given pickle file, or is initialized.
 
 It is based on Stacked de-noising Autoencoder models from the Bengio lab.  
 See http://deeplearning.net/tutorial/SdA.html#sda  

"""
import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from sda import SdA

from extract_datasets import extract_unlabeled_chunkrange
from load_shared import load_data_unlabeled
from tables import openFile

from datetime import datetime

from optparse import OptionParser
import os


def pretrain_SdA(pretraining_epochs=50, pretrain_lr=0.001, batch_size=100):
    """
    
    Pretrain an SdA model for the given number of training epochs.  The model is either initialized from scratch, or 
    is reconstructed from a previously pickled model.

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

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
    output_filename = "stacked_denoising_autoencoder_" + options.arch + "." + day + "." + hour
    output_file = open(output_filename,'w')
    os.chdir(current_dir)    
    print >> output_file, "Run on " + str(datetime.now())    
    
    # Get the training data sample from the input file
    data_set_file = openFile(str(options.inputfile), mode = 'r')
    datafiles = extract_unlabeled_chunkrange(data_set_file, num_files = 25, offset = options.offset)
    train_set_x = load_data_unlabeled(datafiles)
    data_set_file.close()

    # compute number of minibatches for training, validation and testing
    n_train_batches, n_features = train_set_x.get_value(borrow=True).shape
    n_train_batches /= batch_size
    
    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    
    
    # Check if we can restore from a previously trained model,    
    # otherwise construct a new SdA
    if options.restorefile is not None:
        print >> output_file, 'Unpickling the model from %s ...' % (options.restorefile)
        current_dir = os.getcwd()    
        os.chdir(options.dir)         
        f = file(options.restorefile, 'rb')
        sda_model = cPickle.load(f)
        f.close()        
        os.chdir(current_dir)
    else:
        print '... building the model'
        arch_list_str = options.arch.split("-")
        arch_list = [int(item) for item in arch_list_str]
        corruption_list = [options.corruption for i in arch_list]
        sda_model = SdA(numpy_rng=numpy_rng, n_ins=n_features,
              hidden_layers_sizes=arch_list,
              corruption_levels = corruption_list,
              n_outs=-1)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = sda_model.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    print '... pre-training the model'
    start_time = time.clock()
    
    ## Pre-train layer-wise
    corruption_levels = sda_model.corruption_levels
    learning_rates = [pretrain_lr * 10. for i in arch_list]
    learning_rates[0] = pretrain_lr    
    
    for i in xrange(sda_model.n_layers):
                       
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=learning_rates[i],
                         momentum=options.momentum,
                         weight_decay=options.weight_decay))
            print >> output_file, 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print >> output_file, numpy.mean(c)
            
        if options.savefile is not None:
            print >> output_file, 'Pickling the model...'
            current_dir = os.getcwd()    
            os.chdir(options.dir)            
            f = file(options.savefile, 'wb')
            cPickle.dump(sda_model, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()
            os.chdir(current_dir)

    end_time = time.clock()

    print >> output_file, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    output_file.close()


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--dir", dest="dir", help="test output directory")
    parser.add_option("-s","--savefile",dest = "savefile", help = "Save the model to this pickle file", default=None)
    parser.add_option("-r","--restorefile",dest = "restorefile", help = "Restore the model from this pickle file", default=None)
    parser.add_option("-i", "--inputfile", dest="inputfile", help="the data (hdf5 file) prepended with an absolute path")
    parser.add_option("-c", "--corruption", dest="corruption", type="float", help="use this amount of corruption for the dA")
    parser.add_option("-m", "--momentum", dest="momentum", type="float", help="use this amount of momentum when updating", default=0.7)
    parser.add_option("-w", "--weight_decay", dest="weight_decay", type="float", help="use this amount of weight decay for regularizing the weights", default=0.00002)
    parser.add_option("-o", "--offset", dest="offset", type="int", help="use this offset for reading input from the hdf5 file")
    parser.add_option("-a", "--arch", dest="arch", default = "", help="use this dash separated list to specify the architecture of the SdA.  E.g -a 850-400-50")
    (options, args) = parser.parse_args()        
    
    pretrain_SdA()
