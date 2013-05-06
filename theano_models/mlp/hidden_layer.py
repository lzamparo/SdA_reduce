import theano
import theano.tensor as T
import numpy

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation=T.tanh):

        self.input = input

        W_values = numpy.asarray(rng.uniform( \
            low = -numpy.sqrt(6. / (n_in + n_out)), \
            high = numpy.sqrt(6. / (n_in + n_out)), \
            size = (n_in, n_out)), dtype = theano.config.floatX)

        if activation == T.nnet.sigmoid:
            W_values *= 4

        self.W = theano.shared(value=W_values, name = 'W')

        b_values = numpy.zeros((n_out,), dtype = theano.config.floatX)
        self.b = theano.shared(value=b_values, name = 'b')

        self.output = activation(T.dot(input, self.W) + self.b)
        self.params = [self.W, self.b]
        
    def __getstate__():
        return (self.W.get_value(), self.b.get_value(), self.activation, self.input)
    
    def __setstate__(self,state):
        """ Set the parameters of this layer based on the values pulled out of state """
        (W, B, activation, input) = state
        self.W = theano.shared(value=W, name = 'W')
        self.b = theano.shared(value=b, name = 'b')
        self.input = input
        self.output = activation(T.dot(input, self.W) + self.b)
        self.params = [self.W, self.b]
        