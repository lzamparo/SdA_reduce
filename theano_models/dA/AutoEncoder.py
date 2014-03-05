import numpy as np
import theano.tensor as T
from theano import shared, config
from theano.tensor.shared_randomstreams import RandomStreams
import pdb

class AutoEncoder(object):
        
    def __init__(self, numpy_rng=None, theano_rng=None, input=None, n_visible=784, n_hidden=500, 
                 W=None, bhid=None, bvis=None):
        """
            
            A de-noising AutoEncoder class from theano tutorials.  While the constructor 
                :type numpy_rng: numpy.random.RandomState
                :param numpy_rng: number random generator used to generate weights
            
                :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
                :param theano_rng: Theano random generator; if None is given one is generated
                    based on a seed drawn from `rng`
            
                :type input: theano.tensor.TensorType
                :paran input: a symbolic description of the input or None for standalone
                              dA
            
                :type n_visible: int
                :param n_visible: number of visible units
            
                :type n_hidden: int
                :param n_hidden:  number of hidden units
            
                :type W: theano.tensor.TensorType
                :param W: Theano variable pointing to a set of weights that should be
                          shared belong the dA and another architecture; if dA should
                          be standalone set this to None
            
                :type bhid: theano.tensor.TensorType
                :param bhid: Theano variable pointing to a set of biases values (for
                             hidden units) that should be shared belong dA and another
                             architecture; if dA should be standalone set this to None
            
                :type bvis: theano.tensor.TensorType
                :param bvis: Theano variable pointing to a set of biases values (for
                             visible units) that should be shared belong dA and another
                             architecture; if dA should be standalone set this to None
            
                """        
        pdb.set_trace()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        if numpy_rng is None:
            raise AssertionError("numpy_rng cannot be unspecified in AutoEncoder.__init__")
        
        # create a Theano random generator that gives symbolic random values
        if theano_rng is not None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))        
        
        # Pick initial values for W, bvis, bhid based on some formula given by 
        # the Theano dudes.        
        if W is None:      
            initial_W = np.asarray(numpy_rng.uniform(
                low = -4 * np.sqrt(6. / (n_hidden + n_visible)),
                high = 4 * np.sqrt(6. / (n_hidden + n_visible)),
                size = (n_visible, n_hidden)), dtype = config.floatX)
            W = shared(value=initial_W, name='W')
        
        # Bias of the visible units    
        if not bvis:
            bvis = shared(value=np.zeros(n_visible,
                                            dtype = config.floatX), name = 'bvis')
            self.b_prime = bvis
        else:
            self.b_prime = bvis
            
        # Bias of the hidden units    
        if not bhid:
            bhid = shared(value=np.zeros(n_hidden,
                                         dtype = config.floatX), name = 'bhid')
            self.b = bhid
        else:
            self.b = bhid
            
        self.W = W
        
        # Tie the weights of the decoder to the encoder
        self.W_prime = self.W.T
        
        self.theano_rng = theano_rng         
        
        if input is None:
            self.x = T.dmatrix(name='input')
                       
        else:
            self.x = input
            
        self.params = [self.W, self.b, self.b_prime]
        
        
    def get_corrupted_input(self, input, corruption_level):
        """ This function keeps ``1-corruption_level`` entries of the inputs the same
        and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial
  
                this will produce an array of 0s and 1s where 1 has a probability of
                1 - ``corruption_level`` and 0 with ``corruption_level``
        """
        return  self.theano_rng.binomial(size=input.shape, n=1, p=1 - corruption_level) * input    
    
    def get_hidden_values(self, input):
        """ Compute the values of the hidden layer """ 
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)
    
    def get_reconstructed_input(self, hidden):
        """ Compute the reconstructed input given the hidden rep'n """
        raise NotImplementedError(str(type(self))+ " does not implement get_reconstructed_input.")
    
    def get_cost_updates(self, corruption_level, learning_rate):
        """ Compute the reconstruction error over the mini-batched input
       taking into account a certain level of corruption of the input """
       
        raise NotImplementedError(str(type(self))+ " does not implement get_cost_updates.")
        
    
    def __getstate__(self):
        """ Return a tuple of all the important parameters that define this dA """
        return (self.W.get_value(), self.b.get_value(), self.b_prime.get_value(), self.n_visible, self.n_hidden)
    
    def __setstate__(self, state):
        """ Set the state of this dA from values returned from a deserialization process like unpickle. """
        W, b, b_prime, n_visible, n_hidden = state
        self.W = shared(value=W, name='W')
        self.b = shared(value=b, name = 'bvis')
        self.b_prime = shared(value=b_prime, name= 'bhid')
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        numpy_rng = np.random.RandomState(123)
        self.theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.W_prime = self.W.T
        
        self.params = [self.W, self.b, self.b_prime]
        
    def get_params(self):
        """ Return the params of this dA.  This is for pickling testing purposes """
        return self.params
    
    def set_input(self, input):
        """ Set the input for an unpickled dA """
        self.x = input


class BernoulliAutoEncoder(AutoEncoder):
        
    def __init__(self, numpy_rng, theano_rng=None, input=None, n_visible=784, n_hidden=500, 
                 W=None, bhid=None, bvis=None):
        """
            
        A de-noising AutoEncoder with [0,1] inputs and hidden values 
        
            :type numpy_rng: numpy.random.RandomState
            :param numpy_rng: number random generator used to generate weights
        
            :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
            :param theano_rng: Theano random generator; if None is given one is generated
                based on a seed drawn from `rng`
        
            :type input: theano.tensor.TensorType
            :paran input: a symbolic description of the input or None for standalone
                          dA
        
            :type n_visible: int
            :param n_visible: number of visible units
        
            :type n_hidden: int
            :param n_hidden:  number of hidden units
        
            :type W: theano.tensor.TensorType
            :param W: Theano variable pointing to a set of weights that should be
                      shared belong the dA and another architecture; if dA should
                      be standalone set this to None
        
            :type bhid: theano.tensor.TensorType
            :param bhid: Theano variable pointing to a set of biases values (for
                         hidden units) that should be shared belong dA and another
                         architecture; if dA should be standalone set this to None
        
            :type bvis: theano.tensor.TensorType
            :param bvis: Theano variable pointing to a set of biases values (for
                         visible units) that should be shared belong dA and another
                         architecture; if dA should be standalone set this to None
        
        
        """        
        super(BernoulliAutoEncoder,self).__init__(numpy_rng, theano_rng, input, n_visible, n_hidden,
                 W=None, bhid=None, bvis=None)
        self.output = T.nnet.sigmoid(T.dot(input, self.W) + self.b)
        
    @classmethod
    def class_from_values(cls, *args, **kwargs):
        """ This constructor is intended for dynamically constructing a dA layer subclass"""
        #TODO: arg checking in args, kwargs here?  args should just have numpy_rng, kwargs the rest.
        return cls(args,kwargs)        
    
    
    def get_reconstructed_input(self, hidden):
        """ Compute the reconstructed input given the hidden rep'n """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
    
    def get_cost_updates(self, corruption_level, learning_rate):
        """ Compute the reconstruction error over the mini-batched input
        taking into account a certain level of corruption of the input """
        
        x_corrupted = super(BernoulliAutoEncoder,self).get_corrupted_input(self.x, corruption_level)
        y = super(BernoulliAutoEncoder,self).get_hidden_values(x_corrupted)
        z = self.get_reconstructed_input(y)
        
        # Use the cross entropy loss
        L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
            
        cost = T.mean(L)
        
        # compute the gradients of the cost of the dA w.r.t the params
        gparams = T.grad(cost, self.params)
        
        # populate the list of updates to each param
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))
            
        return (cost, updates)
    
    
class GaussianAutoEncoder(AutoEncoder):
        
    def __init__(self, numpy_rng, theano_rng=None, input=None, n_visible=784, n_hidden=500, 
                 W=None, bhid=None, bvis=None):
        """
            
        A de-noising AutoEncoder with Gaussian visible units
            
            :type numpy_rng: numpy.random.RandomState
            :param numpy_rng: number random generator used to generate weights
        
            :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
            :param theano_rng: Theano random generator; if None is given one is generated
                based on a seed drawn from `rng`
        
            :type input: theano.tensor.TensorType
            :paran input: a symbolic description of the input or None for standalone
                          dA
        
            :type n_visible: int
            :param n_visible: number of visible units
        
            :type n_hidden: int
            :param n_hidden:  number of hidden units
        
            :type W: theano.tensor.TensorType
            :param W: Theano variable pointing to a set of weights that should be
                      shared belong the dA and another architecture; if dA should
                      be standalone set this to None
        
            :type bhid: theano.tensor.TensorType
            :param bhid: Theano variable pointing to a set of biases values (for
                         hidden units) that should be shared belong dA and another
                         architecture; if dA should be standalone set this to None
        
            :type bvis: theano.tensor.TensorType
            :param bvis: Theano variable pointing to a set of biases values (for
                         visible units) that should be shared belong dA and another
                         architecture; if dA should be standalone set this to None
                         
            :type loss: string
            :param loss: specify the type of loss function to use when computing the loss
            in get_cost_updates(...).  Currently defined values are 'xent' for cross-entropy, 
            'squared' for squared error.
            
            :type decoder: string
            :param decoder: specify the decoding function to use when computing get_cost_updates(...).  
            Currently defined values are 'sigmoid' for sigmoid, 'linear' for linear. 
        
        
        """        
        super(GaussianAutoEncoder,self).__init__(numpy_rng, theano_rng, input, n_visible, n_hidden, W, bhid, bvis)
        self.output = T.nnet.sigmoid(T.dot(input, self.W) + self.b)    

    @classmethod
    def class_from_values(cls, *args, **kwargs):
        """ This constructor is intended for dynamically constructing a dA layer subclass"""
        #TODO: arg checking in args, kwargs here?  args should just have numpy_rng, kwargs the rest.
        return cls(args,kwargs)        
    
    def get_reconstructed_input(self, hidden):
        """ Use a linear decoder to compute the reconstructed input given the hidden rep'n """
        return T.dot(hidden, self.W_prime) + self.b_prime
    
    def get_cost_updates(self, corruption_level, learning_rate):
        """ Compute the reconstruction error over the mini-batched input
       taking into account a certain level of corruption of the input """
        x_corrupted = super(GaussianAutoEncoder,self).get_corrupted_input(self.x, corruption_level)
        y = super(GaussianAutoEncoder,self).get_hidden_values(x_corrupted)
        z = self.get_reconstructed_input(y)
        
        # Take the sum over columns
        # Use the squared error loss function
        L = T.sum((self.x - z) **2, axis = 1)
            
        cost = T.mean(L)
        
        # compute the gradients of the cost of the dA w.r.t the params
        gparams = T.grad(cost, self.params)
        
        # populate the list of updates to each param
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))
            
        return (cost, updates)
    
    
class ReluAutoEncoder(AutoEncoder):        
    def __init__(self, numpy_rng, theano_rng=None, input=None, n_visible=784, n_hidden=500, 
                 W=None, bhid=None, bvis=None):
        """
            
        A de-noising AutoEncoder with ReLu visible units
            
            :type numpy_rng: numpy.random.RandomState
            :param numpy_rng: number random generator used to generate weights
        
            :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
            :param theano_rng: Theano random generator; if None is given one is generated
                based on a seed drawn from `rng`
        
            :type input: theano.tensor.TensorType
            :paran input: a symbolic description of the input or None for standalone
                          dA
        
            :type n_visible: int
            :param n_visible: number of visible units
        
            :type n_hidden: int
            :param n_hidden:  number of hidden units
        
            :type W: theano.tensor.TensorType
            :param W: Theano variable pointing to a set of weights that should be
                      shared belong the dA and another architecture; if dA should
                      be standalone set this to None
        
            :type bhid: theano.tensor.TensorType
            :param bhid: Theano variable pointing to a set of biases values (for
                         hidden units) that should be shared belong dA and another
                         architecture; if dA should be standalone set this to None
        
            :type bvis: theano.tensor.TensorType
            :param bvis: Theano variable pointing to a set of biases values (for
                         visible units) that should be shared belong dA and another
                         architecture; if dA should be standalone set this to None
                         
            :type loss: string
            :param loss: specify the type of loss function to use when computing the loss
            in get_cost_updates(...).  Currently defined values are 'xent' for cross-entropy, 
            'squared' for squared error.
            
            :type decoder: string
            :param decoder: specify the decoding function to use when computing get_cost_updates(...).  
            Currently defined values are 'sigmoid' for sigmoid, 'linear' for linear. 
        
        
        """ 
        
        pdb.set_trace()
        super(ReluAutoEncoder,self).__init__(numpy_rng, theano_rng, input, n_visible, n_hidden, W, bhid, bvis)
        self.output = T.maximum(T.dot(input, self.W) + self.b, 0)
        
    @classmethod
    def class_from_values(cls, *args, **kwargs):
        """ This constructor is intended for dynamically constructing a dA layer subclass 
            Args that get specified through this version of the constructor: numpy_rng, theano_rng, input, n_visible, n_hidden
        """
        pdb.set_trace()
        return cls(numpy_rng=kwargs['numpy_rng'], theano_rng=kwargs['theano_rng'], input=kwargs['input'], n_visible=kwargs['n_visible'], n_hidden=kwargs['n_hidden'])        

    def get_reconstructed_input(self, hidden):
        """ Use a linear decoder to compute the reconstructed input given the hidden rep'n """
        return T.dot(hidden, self.W_prime) + self.b_prime
    
    def get_hidden_values(self,input):
        """ Apply ReLu elementwise to the input """
        return T.maximum(T.dot(self.W, input) + self.b, 0)
    
    def get_cost_updates(self, corruption_level, learning_rate):
        """ Compute the reconstruction error over the mini-batched input
       taking into account a certain level of corruption of the input """
        x_corrupted = super(ReluAutoEncoder,self).get_corrupted_input(self.x, corruption_level)
        y = super(ReluAutoEncoder,self).get_hidden_values(x_corrupted)
        z = self.get_reconstructed_input(y)
        
        # Take the sum over columns
        # Use the squared error loss function
        L = T.sum((self.x - z) **2, axis = 1)
            
        cost = T.mean(L)
        
        # compute the gradients of the cost of the dA w.r.t the params
        gparams = T.grad(cost, self.params)
        
        # populate the list of updates to each param
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))
            
        return (cost, updates)