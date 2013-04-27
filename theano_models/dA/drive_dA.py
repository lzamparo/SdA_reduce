# try out the denoising Autoencoder on MNIST 

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import load_data
from AutoEncoder import AutoEncoder

from datetime import datetime

from optparse import OptionParser
import os


def drive_dA(learning_rate=0.1, training_epochs=15,
            dataset='../data/mnist.pkl.gz',
            batch_size=20):
    """
        This demo is tested on MNIST
    
        :type learning_rate: float
        :param learning_rate: learning rate used for training the DeNosing
                              AutoEncoder
    
        :type training_epochs: int
        :param training_epochs: number of epochs used for training
    
        :type dataset: string
        :param dataset: path to the picked dataset
    
    """
    parser = OptionParser()
    parser.add_option("-d", "--dir", dest="dir", help="test output directory")
    parser.add_option("-c", "--corruption", dest="corruption", help="use this amount of corruption for the denoising AE")
    
    (options, args) = parser.parse_args()    

    current_dir = os.getcwd()
    
    os.chdir(options.dir)
    today = datetime.today()
    day = str(today.date())
    hour = str(today.time())
    output_filename = "denoising_autoencoder_mnist." + day + "." + hour
    output_file = open(output_filename,'w')
    
    print >> output_file, "Run on " + str(datetime.now())    
    
    os.chdir(current_dir)
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    
    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(numpy_rng=rng, theano_rng=theano_rng, input=x,
            n_visible=28 * 28, n_hidden=500)

    cost, updates = da.get_cost_updates(corruption_level=0.,
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
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print >> output_file, 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> output_file, ('The 0 corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))    
    
            
    ##########
    # Build the model, with corruption #
    ##########
    
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(numpy_rng=rng, theano_rng=theano_rng, input=x,
            n_visible=28 * 28, n_hidden=500)

    cost, updates = da.get_cost_updates(corruption_level=options.corruption,
                                        learning_rate=learning_rate)

    train_da = theano.function([index], cost, updates=updates,
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
            c.append(train_da(batch_index))

        print >> output_file, 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> output_file, ('The ' + str(options.corruption) + '% corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))

    output_file.close()
    
if __name__ == '__main__':
    drive_dA()    