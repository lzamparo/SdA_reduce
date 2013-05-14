import numpy

import theano
import theano.tensor as T
from AutoEncoder import AutoEncoder
from theano.tensor.shared_randomstreams import RandomStreams

from extract_datasets import extract_labeled_chunkrange
from load_shared import load_data_labeled
from tables import *

import os
import sys
import time
from datetime import datetime
from optparse import OptionParser


def test_pickled_sqe_dA(learning_rate=0.001,            
            pickle_file='/scratch/z/zhaolei/lzamparo/gpu_tests/dA_results/dA_sqe_pickle.save',
            corruption=0.1,
            training_epochs=3,
            batch_size=20):
    """ Test creating a dA model from scratch, training for a set number of epochs, pickle the model, unpickle, continue. """   

    current_dir = os.getcwd()    

    os.chdir(options.dir)
    today = datetime.today()
    day = str(today.date())
    hour = str(today.time())
    output_filename = "test_dA_squarederror_pickle." + day + "." + hour
    output_file = open(output_filename,'w')
    
    print >> output_file, "Run on " + str(datetime.now())    
    
    os.chdir(current_dir)
    
    data_set_file = openFile(str(options.inputfile), mode = 'r')
    datafiles, labels = extract_labeled_chunkrange(data_set_file, num_files = 10)
    datasets = load_data_labeled(datafiles, labels)
    train_set_x, train_set_y = datasets[0]
    data_set_file.close()

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_cols = train_set_x.get_value(borrow=True).shape[1]	

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data matrix
    
    ####################################
    # BUILDING THE MODEL #
    ####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = AutoEncoder(numpy_rng=rng, theano_rng=theano_rng, input=x,
            n_visible=n_cols, n_hidden=1000, loss='squared')

    cost, updates = da.get_cost_updates(corruption_level=float(options.corruption),
                                        learning_rate=learning_rate)

    train_da = theano.function([index], cost, updates=updates,
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
            c.append(train_da(batch_index))

        print >> output_file, 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> output_file, ('The 0 corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))  
    
    ############
    # Pickle #
    ############
    
    f = file(pickle_file, 'wb')
    cPickle.dump(da, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
        
    ############
    # Unpickle the model, try to recover #
    ############
    
    f = file(pickle_file, 'rb')
    pickled_dA = cPickle.load(f)
    f.close()    
    
    
    ############
    # Resume training #
    ############    
    
    cost, updates = pickled_dA.get_cost_updates(corruption_level=float(options.corruption),
                                            learning_rate=learning_rate)
    
    train_da = theano.function([index], cost, updates=updates,
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
            c.append(train_da(batch_index))

        print >> output_file, 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> output_file, ('The 0 corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))
    
    output_file.close()
    

def test_unpickled_sqe_dA(learning_rate=0.001,
            dataset='../data/mnist.pkl.gz',
            pickle_file='/scratch/z/zhaolei/lzamparo/gpu_tests/dA_results/dA_sqe_pickle.save',
            corruption=0.1,
            training_epochs=3,
            batch_size=20):
    """ Unpickle the da in the given file, train for a few more epochs """
    
    current_dir = os.getcwd()    
    
    os.chdir(options.dir)
    today = datetime.today()
    day = str(today.date())
    hour = str(today.time())
    output_filename = "test_dA_squarederror_unpickle." + day + "." + hour
    output_file = open(output_filename,'w')
    
    print >> output_file, "Run on " + str(datetime.now())    
    
    os.chdir(current_dir)
    
    data_set_file = openFile(str(options.inputfile), mode = 'r')
    datafiles, labels = extract_labeled_chunkrange(data_set_file, num_files = 10)
    datasets = load_data_labeled(datafiles, labels)
    train_set_x, train_set_y = datasets[0]
    data_set_file.close()

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_cols = train_set_x.get_value(borrow=True).shape[1]	

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data matrix    
    
    ############
    # Unpickle the model, try to recover #
    ############
    
    f = file(pickle_file, 'rb')
    pickled_dA = cPickle.load(f)
    f.close()    
    
    ############
    # Resume training #
    ############    
    
    cost, updates = pickled_dA.get_cost_updates(corruption_level=float(options.corruption),
                                            learning_rate=learning_rate)
    
    train_da = theano.function([index], cost, updates=updates,
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
            c.append(train_da(batch_index))

        print >> output_file, 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> output_file, ('The 0 corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.)) 
    output_file.close()

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-d", "--dir", dest="dir", help="test output directory")
    parser.add_option("-c", "--corruption", dest="corruption", help="use this amount of corruption for the denoising AE")
    parser.add_option("-i", "--inputfile", dest="inputfile", help="the hdf5 filename as an absolute pathname")
    (options, args) = parser.parse_args()   
    
    test_pickled_sqe_dA();
    test_unpickled_sqe_dA();