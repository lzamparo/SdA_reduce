import theano
import theano.tensor as T
import numpy

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation=T.nnet.sigmoid):

        self.input = input
        
        # Recipe for initializing W for tanh activation layers
        W_values = numpy.asarray(rng.uniform( \
            low = -numpy.sqrt(6. / (n_in + n_out)), \
            high = numpy.sqrt(6. / (n_in + n_out)), \
            size = (n_in, n_out)), dtype = theano.config.floatX)

        if activation == T.nnet.sigmoid:
            W_values *= 4

        self.W = theano.shared(value=W_values, name = 'W')

        b_values = numpy.zeros((n_out,), dtype = theano.config.floatX)
        self.b = theano.shared(value=b_values, name = 'b')
        self._outsize = n_out

        self.output = activation(T.dot(input, self.W) + self.b)
        self.params = [self.W, self.b]
        
    def __getstate__(self):
        """ Return the weight matrix and bias parameters. """
        return (self.W.get_value(), self.b.get_value(), self._outsize)
    
    def __setstate__(self,state):
        """ Set the parameters of this layer based on the values pulled out of state. """
        (W, b, outsize) = state
        self.W = theano.shared(value=W, name = 'W')
        self.b = theano.shared(value=b, name = 'b')
        self._outsize = outsize
               
    def reconstruct_state(self, input, activation=T.nnet.sigmoid):
        """ Set up the symbolic input, outputs as in the constructor. """
        self.input = input
        self.output = activation(T.dot(input, self.W) + self.b)
        self.params = [self.W, self.b]
        
    def get_params(self):
        """ Return the params of this MLP.  This is for pickling testing purposes """
        return self.params    
    
        