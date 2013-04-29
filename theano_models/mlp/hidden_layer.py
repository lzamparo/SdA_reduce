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