import numpy as np

import theano
import theano.tensor as T

# test for theano tensor.maximum

data = theano.shared(np.eye(50, dtype=theano.config.floatX), name="testdata")
bias = theano.shared(np.zeros(50, dtype=theano.config.floatX), name="bias", strict=False, allow_downcast=None)

init_W = np.random.randn(50*50).reshape((50,50))
init_W.dtype = theano.config.floatX

W = theano.shared(init_W, name="W")

index = T.lscalar()
d = T.fmatrix(name='d')

#  self.x: train_set_x[index * batch_size: (index + 1) * batch_size] 

mat_mult = T.dot(d,W) + bias
test_dot = theano.function([index],mat_mult, givens = {d: data[index : (index + 2),:]})

relu_act = T.maximum(T.dot(d, W) + bias, 0.0)
test_max = theano.function([index],relu_act, givens = {d: data[index : (index + 2),:]})

for i in xrange(48):
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
        
        

