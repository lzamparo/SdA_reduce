import numpy as np

import theano
import theano.tensor as T

# test for theano tensor.maximum

data = theano.shared(np.eye(50, dtype=theano.config.floatX), name="testdata")
bias = theano.shared(np.zeros(50, dtype=theano.config.floatX), name="bias", strict=False, allow_downcast=None)

init_W = np.random.randn(50*50).reshape((50,50))
init_W.dtype = theano.config.floatX

W = theano.shared(init_W, name="W")

index = T.scalar()
d = T.row(name='d', dtype=theano.config.floatX)

test_dot = theano.function([index],T.dot(d,W) + bias, givens = {d: data[index,:]})

test_max = theano.function([index],T.maximum(T.dot(d, W) + bias, 0.0), givens = {d: data[index,:]})

for i in xrange(50):
    out = test_max(i)
    mult = test_dot(i)
    if np.isnan(np.sum(out)):
        print "Got NaNs in test with T.maximum"
    else:
        print "No NaNs in test with T.maximum"
    if np.isnan(np.sum(mult)):
        print "Got NaNs with T.dot(d,W)"
    else:
        print "No NaNs in test with T.dot(d,W)"
        
        

