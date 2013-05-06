import cPickle
import gzip
import sys

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from mlp.logistic_sgd import load_data
from AutoEncoder import AutoEncoder

    
def test_pickled_dA(learning_rate=0.1,
            dataset='../data/mnist.pkl.gz',
            pickle_file='/scratch/z/zhaolei/lzamparo/gpu_tests/dA_results/dA_pickle.save',
            corruption=0.1,
            training_epochs=3,
            batch_size=20):
    """
        Test pickling, unpickling code for the dA class.  Start up a model, train, pickle, unpickle, and continue to train  
        
    """
    
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    
    ####################################
    # Build the model #
    ####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = AutoEncoder(numpy_rng=rng, theano_rng=theano_rng, input=x,
            n_visible=28 * 28, n_hidden=500, loss='xent')

    cost, updates = da.get_cost_updates(corruption_level=0., learning_rate=learning_rate)

    train_da = theano.function([index], cost, updates=updates,
         givens={x: train_set_x[index * batch_size:
                                (index + 1) * batch_size]})

    ############
    # Train the model for 3 epochs #
    ############

    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through training set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))
    
    ############
    # Pickle the model #
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
    # Compare the two models #
    ###########
    dA_params = da.get_params()
    pickled_params = pickled_dA.get_params()
    
    if not numpy.allclose(dA_params[0].get_value(), pickled_params[0].get_value()):
        print "numpy says that Ws are not close"
    if not numpy.allclose(dA_params[1].get_value(), pickled_params[1].get_value()):
        print "numpy says that the bvis are not close"
    if not numpy.allclose(dA_params[2].get_value(), pickled_params[2].get_value()):
        print "numpy says that the bhid are not close"
    

    ############
    # Compare the two models #
    ##########    
    pickled_dA.set_input(x)
    cost, updates = pickled_dA.get_cost_updates(corruption_level=0.1, learning_rate=learning_rate)
    
    pickle_train_da = theano.function([index], cost, updates=updates,
         givens={x: train_set_x[index * batch_size:
                                (index + 1) * batch_size]})
    
    ############
    # Train the model for 3 epochs #
    ############

    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through training set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))    
    
    print "Passed create, pickle, unpickle, train test"
            

def test_continue_pickled_dA(learning_rate=0.1,
            dataset='../data/mnist.pkl.gz',
            pickle_file='/scratch/z/zhaolei/lzamparo/gpu_tests/dA_results/dA_pickle.save',
            corruption=0.1,
            training_epochs=3,
            batch_size=20):
    """
        Test unpickling a dA, followed by training for some more epochs.  
        Unpickle, set up and train
    
    """  
        
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    
    ####################################
    # Build the model #
    ####################################    
    
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images    
    
    ############
    # Unpickle the model, try to recover #
    ############
    
    f = file(pickle_file, 'rb')
    unpickled_dA = cPickle.load(f)
    f.close()    
    
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    
    unpickled_dA.sett_input(x)
    
    cost, updates = unpickled_dA.get_cost_updates(corruption_level=0., learning_rate=learning_rate)
    
    train_da = theano.function([index], cost, updates=updates,
         givens={x: train_set_x[index * batch_size:
                                (index + 1) * batch_size]})

    ############
    # Train the model for 3 more epochs #
    ############

    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through training set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))    
    

    print "passed unpickled dA test"    
    
if __name__ == '__main__':
    test_pickled_dA()
    test_continue_pickled_dA()
