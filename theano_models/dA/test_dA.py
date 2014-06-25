# Experiment script to compare ReLU dAs versus Gaussian-Bernoulli dAs
# Train each with varying amounts of noise, 
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from AutoEncoder import AutoEncoder
from AutoEncoder import ReluAutoEncoder
from AutoEncoder import GaussianAutoEncoder

from extract_datasets import extract_labeled_chunkrange
from load_shared import load_data_labeled
from tables import *

import os
import sys
import time
from datetime import datetime
from optparse import OptionParser

from sklearn.preprocessing import scale


def test_gb_dA(options,learning_rate=0.001, training_epochs=10, hidden_layer_size = 50,
             batch_size=50):
    """
        Perform some simple tests with a GB dA is (with foci data) 
    
        :type options: opt_parse options
        :param options: opt_parse options
    
        :type learning_rate: float
        :param learning_rate: learning rate used for training the De-noising
                              AutoEncoder
    
        :type training_epochs: int
        :param training_epochs: number of epochs used for training
        
        :type hidden_layer_size: int
        :param hidden_layer_size: size of the hidden layer for this GB dA
    
        :type batch_size: int
        :param batch_size: size of each minibatch
    
    """
        

    current_dir = os.getcwd()    

    os.chdir(options.dir)
    today = datetime.today()
    day = str(today.date())
    hour = str(today.time())
    corruptn = str(options.corruption)
    output_filename = "test_gb_da." + "corruption_" + corruptn + "_" + day + "." + hour
    output_file = open(output_filename,'w')
    
    print >> output_file, "Run on " + str(datetime.now())    
    
    os.chdir(current_dir)
    
    data_set_file = openFile(str(options.inputfile), mode = 'r')
    datafiles, labels = extract_labeled_chunkrange(data_set_file, num_files = 10)
    datasets = load_data_labeled(datafiles, labels)
    train_set_x, train_set_y = datasets[0]
    data_set_file.close()
    
    # Get the data on the host for comparison.
    #h_train = scale(datafiles[:,5:612])
    #h_train = h_train[:train_set_x.get_value(borrow=True).shape(0),:]
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_cols = train_set_x.get_value(borrow=True).shape[1]	

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data matrix
    
    ##################################
    # Build the GaussianBernoulli dA #
    ##################################

    rng = numpy.random.RandomState(2345)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = GaussianAutoEncoder(numpy_rng=rng, theano_rng=theano_rng, input=x,
            n_visible=n_cols, n_hidden=hidden_layer_size, sparse_init=10)

    cost, hiddens, reconstructed, updates = da.get_cost_updates_debug(corruption_level=options.corruption,
                                        learning_rate=learning_rate)

    debug_train = theano.function([index], [cost, hiddens, reconstructed], updates=updates,
         givens={x: train_set_x[index * batch_size:
                                (index + 1) * batch_size]})
    
    start_time = time.clock()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through training set
        c = []
        for batch_index in xrange(n_train_batches):
            error, y, z = debug_train(batch_index)
            if numpy.isnan(numpy.sum(z)) or numpy.isnan(numpy.sum(y)):
                print >> output_file, "number of nan values in hidden reps: %d " % numpy.sum(numpy.isnan(y))
                print >> output_file, "number of nan values in reconstructed vals: %d " % numpy.sum(numpy.isnan(z))
            c.append(error)

        print >> output_file, 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> output_file, ('The ' + str(options.corruption) + ' corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))    
    
    output_file.close()      


def test_relu_dA(options, learning_rate=0.001, training_epochs=10, hidden_layer_size = 50,
             batch_size=50):
    """
        Perform some simple tests with a GB dA is (with foci data) 
    
        :type options: opt_parse options
        :param options: opt_parse options
    
        :type learning_rate: float
        :param learning_rate: learning rate used for training the De-noising
                              AutoEncoder
    
        :type training_epochs: int
        :param training_epochs: number of epochs used for training
        
        :type hidden_layer_size: int
        :param hidden_layer_size: size of the hidden layer for this GB dA
    
        :type batch_size: int
        :param batch_size: size of each minibatch
    
    """    
    ##########
    # Build the ReLU dA
    ##########
    output_filename = "relu_da." + "corruption_" + corruptn + "_" + day + "." + hour
    
    current_dir = os.getcwd()    
    os.chdir(options.dir)    
    output_file = open(output_filename,'w')
    os.chdir(current_dir)
    print >> output_file, "Run on " + str(datetime.now())     
    
    os.chdir(current_dir)
        
    data_set_file = openFile(str(options.inputfile), mode = 'r')
    datafiles, labels = extract_labeled_chunkrange(data_set_file, num_files = 10)
    datasets = load_data_labeled(datafiles, labels)
    train_set_x, train_set_y = datasets[0]
    data_set_file.close()    
    
    # Get the data on the host for comparison.
    #h_train = scale(datafiles[:,5:612])
    #h_train = h_train[:train_set_x.get_value(borrow=True).shape(0),:]    
    
    rng = numpy.random.RandomState(6789)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
 
    da = ReluAutoEncoder(numpy_rng=rng, theano_rng=theano_rng, input=x,
                n_visible=n_cols, n_hidden=hidden_layer_size, sparse_init=10)    

    cost, hiddens, reconstructed, updates = da.get_cost_updates_debug(corruption_level=float(options.corruption),
                                        learning_rate=learning_rate)

    debug_train = theano.function([index], [cost, hiddens, reconstructed], updates=updates,
         givens={x: train_set_x[index * batch_size:
                                (index + 1) * batch_size]})

    start_time = time.clock()
    
    ##########
    # Train the model
    ##########
    
    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            error, y, z = debug_train(batch_index)
            if numpy.isnan(numpy.sum(z)) or numpy.isnan(numpy.sum(y)):
                print >> output_file, "number of nan values in hidden reps: %d " % numpy.sum(numpy.isnan(y))
                print >> output_file, "number of nan values in reconstructed vals: %d " % numpy.sum(numpy.isnan(z))
            c.append(error)

        print >> output_file, 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> output_file, ('The ' + str(options.corruption) + ' corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))

    output_file.close()
    
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--dir", dest="dir", help="test output directory")
    parser.add_option("-c", "--corruption", dest="corruption", help="use this amount of corruption for the denoising AE", type="float")
    parser.add_option("-i", "--inputfile", dest="inputfile", help="the hdf5 filename as an absolute pathname")
    (options, args) = parser.parse_args()    
    test_gb_dA(options) 
    test_relu_dA(options) 
