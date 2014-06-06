import numpy as np
import theano.tensor as T
from theano import shared, config
from theano.tensor.shared_randomstreams import RandomStreams


class AutoEncoder(object):
        
    def __init__(self, numpy_rng=None, theano_rng=None, input=None, n_visible=784, n_hidden=500, 
                 W=None, bhid=None, bvis=None, W_name=None, bvis_name=None, bhid_name=None):
        """ A de-noising AutoEncoder class from theano tutorials.  
        
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
                      shared Theano variables connecting the visible and hidden layers.
                      
                :type bhid: theano.tensor.TensorType
                :param bhid: Theano variable pointing to a set of biases values (for
                             hidden units).
            
                :type bvis: theano.tensor.TensorType
                :param bvis: Theano variable pointing to a set of biases values (for
                             visible units).
                             
                :type W_name: string
                :param W_name: name to be assigned to the W matrix.
            
                :type bvis_name: string
                :param bvis_name: name to be assigned to the b vector for the visible units.
                
                :type bhid_name: string
                :param bhid_name: name to be assigned to the b vector for the hidden units.
            
                """        
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        if numpy_rng is None:
            raise AssertionError("numpy_rng cannot be unspecified in AutoEncoder.__init__")
        
        # create a Theano random generator that gives symbolic random values
        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))        
        
        if W_name is None:
            W_name = 'W'
        if bvis_name is None:
            bvis_name = 'bvis'
        if bhid_name is None:
            bhid_name = 'bhid'
        
        # Pick initial values for W, bvis, bhid based on some formula given by 
        # a paper by Glorot and Bengio (AISTATS2010).        
        # N.B. this is only appropriate for sigmoid or tanh units.  ReLU units won't work well 
        # here since many of them will be < 0 and thus shut off.
        
        if not W:
            initial_W = np.asarray(numpy_rng.uniform(
                low = -4 * np.sqrt(6. / (n_hidden + n_visible)),
                high = 4 * np.sqrt(6. / (n_hidden + n_visible)),
                size = (n_visible, n_hidden)), dtype = config.floatX)
            W = shared(value=initial_W, name=W_name)
        
        self.W = W
        # Tie the weights of the decoder to the encoder
        self.W_prime = self.W.T        

        # Bias of the visible units    
        if not bvis:
            bvis = shared(value=np.zeros(n_visible,
                                            dtype = config.floatX), name = bvis_name)
            
        self.b_prime = bvis
            
        # Bias of the hidden units    
        if not bhid:
            bhid = shared(value=np.zeros(n_hidden,
                                         dtype = config.floatX), name = bhid_name)
        self.b = bhid
        
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
        raise NotImplementedError(str(type(self))+ " does not implement get_hidden_values.")
    
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
        
    def dropout_from_layer(layer, p):
        """ Apply masking noise to the hidden (i.e output) layer for this dA.
        
        :type layer: theano.shared 
        :param layer: number random generator used to generate weights
        
        :type p: float
        :param p: retain each unit in this layer with probability p """
        
        mask = self.theano_rng.binomial(size=layer.shape, n=1, p=p) * layer
        # According to https://github.com/mdenil/dropout/blob/master/mlp.py:
        # The cast is important because int * float32 = float64, 
        # which pulls things off the gpu.  He apparently has not heard of allow.downcast.
        return layer * T.cast(mask, theano.config.floatX)    


class BernoulliAutoEncoder(AutoEncoder):
        
    def __init__(self, numpy_rng, theano_rng=None, input=None, n_visible=784, n_hidden=500, 
                 W=None, bhid=None, bvis=None, W_name=None, bvis_name=None, bhid_name=None):
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
                      shared Theano variables connecting the visible and hidden layers.
                      
        
            :type bhid: theano.tensor.TensorType
            :param bhid: Theano variable pointing to a set of biases values (for
                         hidden units).
        
            :type bvis: theano.tensor.TensorType
            :param bvis: Theano variable pointing to a set of biases values (for
                         visible units).
                         
            :type W_name: string
            :param W_name: name to be assigned to the W matrix.
            
            :type bvis_name: string
            :param bvis_name: name to be assigned to the bvis vector.
            
            :type bhid_name: string
            :param bhid_name: name to be assigned to the bhid vector.
        
        
        """        
        super(BernoulliAutoEncoder,self).__init__(numpy_rng, theano_rng, input, n_visible, n_hidden, W, bhid, bvis, W_name, bvis_name, bhid_name)
        self.output = T.nnet.sigmoid(T.dot(input, self.W) + self.b)
        
    @classmethod
    def class_from_values(cls, *args, **kwargs):
        """ This constructor is intended for dynamically constructing a dA layer subclass 
            Args that always get specified in this constructor: numpy_rng, theano_rng, input, n_visible, n_hidden, W_name, bvis_name, bhid_name
            Args that *sometimes* get specified in this constructor: W, bvis, bhid
        """  
        
        keys = kwargs.keys()
        if 'W' not in keys:
            kwargs['W'] = None
        if 'bhid' not in keys:
            kwargs['bhid'] = None
        if 'bvis' not in keys:
            kwargs['bvis'] = None       
        
        return cls(numpy_rng=kwargs['numpy_rng'], theano_rng=kwargs['theano_rng'], input=kwargs['input'], 
                   n_visible=kwargs['n_visible'], n_hidden=kwargs['n_hidden'], W=kwargs['W'],
                   bhid=kwargs['bhid'], bvis=kwargs['bvis'], W_name=kwargs['W_name'], 
                   bvis_name=kwargs['bvis_name'], bhid_name=kwargs['bhid_name'])        
    
    def get_hidden_values(self, input):
        """ Compute the values of the hidden layer """    
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)
    
    def get_reconstructed_input(self, hidden):
        """ Compute the reconstructed input given the hidden rep'n """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
    
    def get_cost_updates(self, corruption_level, learning_rate):
        """ Compute the reconstruction error over the mini-batched input
        taking into account a certain level of corruption of the input """
        
        x_corrupted = super(BernoulliAutoEncoder,self).get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(x_corrupted)
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
    
    def get_cost_gparams(self, corruption_level, learning_rate):
        """ Compute the reconstruction error over the mini-batched input (with corruption)
       
       But instead of returning a list of tuples (updates) were the rval has the form of an update to 
       a theano.tensor variable (param, update_value), return instead (param, gparam)."""
        x_corrupted = super(BernoulliAutoEncoder,self).get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(x_corrupted)
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
            updates.append((param, learning_rate * gparam))
            
        return (cost, updates)    
    
    
class GaussianAutoEncoder(AutoEncoder):
        
    def __init__(self, numpy_rng, theano_rng=None, input=None, n_visible=784, n_hidden=500, 
                 W=None, bhid=None, bvis=None, W_name=None, bvis_name=None, bhid_name=None):
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
                      shared Theano variables connecting the visible and hidden layers.
                      
        
            :type bhid: theano.tensor.TensorType
            :param bhid: Theano variable pointing to a set of biases values (for
                         hidden units).
        
            :type bvis: theano.tensor.TensorType
            :param bvis: Theano variable pointing to a set of biases values (for
                         visible units).
                         
            :type W_name: string
            :param W_name: name to be assigned to the W matrix.
            
            :type bvis_name: string
            :param bvis_name: name to be assigned to the bvis vector.
            
            :type bhid_name: string
            :param bhid_name: name to be assigned to the bhid vector.
        
        
        """
               
        super(GaussianAutoEncoder,self).__init__(numpy_rng, theano_rng, input, n_visible, n_hidden, W, bhid, bvis, W_name, bvis_name, bhid_name)
        self.output = T.nnet.sigmoid(T.dot(input, self.W) + self.b)    

    @classmethod
    def class_from_values(cls, *args, **kwargs):
        """ This constructor is intended for dynamically constructing a dA layer subclass 
            Args that always get specified through this constructor: numpy_rng, theano_rng, input, n_visible, n_hidden, W_name, bvis_name, bhid_name.
            Args that *might* be specified: W, bhid, bvis.
        """
        
        keys = kwargs.keys()
        if 'W' not in keys:
            kwargs['W'] = None
    
        if 'bhid' not in keys:
            kwargs['bhid'] = None
            
        if 'bvis' not in keys:
            kwargs['bvis']=None
            
            
        return cls(numpy_rng=kwargs['numpy_rng'], theano_rng=kwargs['theano_rng'], input=kwargs['input'], 
                   n_visible=kwargs['n_visible'], n_hidden=kwargs['n_hidden'],W=kwargs['W'],
                   bhid=kwargs['bhid'], bvis=kwargs['bvis'],W_name=kwargs['W_name'], 
                   bvis_name=kwargs['bvis_name'], bhid_name=kwargs['bhid_name'])        
    
    def get_hidden_values(self, input):
        """ Compute the values of the hidden layer """    
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)    
    
    def get_reconstructed_input(self, hidden):
        """ Use a linear decoder to compute the reconstructed input given the hidden rep'n """
        return T.dot(hidden, self.W_prime) + self.b_prime
    
    def get_cost_updates(self, corruption_level, learning_rate):
        """ Compute the reconstruction error over the mini-batched input
       taking into account a certain level of corruption of the input """
        x_corrupted = super(GaussianAutoEncoder,self).get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(x_corrupted)
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
    
    def get_cost_gparams(self, corruption_level, learning_rate):
        """ Compute the reconstruction error over the mini-batched input (with corruption)
    
    But instead of returning a list of tuples (updates) were the rval has the form of an update to 
    a theano.tensor variable (param, update_value), return instead (param, gparam)."""
        x_corrupted = super(GaussianAutoEncoder,self).get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(x_corrupted)
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
            updates.append((param, learning_rate * gparam))
            
        return (cost, updates)    
    
    
class ReluAutoEncoder(AutoEncoder):        
    def __init__(self, numpy_rng, theano_rng=None, input=None, n_visible=784, n_hidden=500, 
                 W=None, bhid=None, bvis=None, W_name=None, bvis_name=None, bhid_name=None):
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
                      shared Theano variables connecting the visible and hidden layers.
                      
        
            :type bhid: theano.tensor.TensorType
            :param bhid: Theano variable pointing to a set of biases values (for
                         hidden units).
        
            :type bvis: theano.tensor.TensorType
            :param bvis: Theano variable pointing to a set of biases values (for
                         visible units).
                         
            :type W_name: string
            :param W_name: name to be assigned to the W matrix.
            
            :type bvis_name: string
            :param bvis_name: name to be assigned to the bvis vector.
            
            :type bhid_name: string
            :param bhid_name: name to be assigned to the bhid vector.
            
        
        """        
        
        # ReLU units require a different weight matrix initialization scheme
        if W_name is None:
            W_name = 'W'
        if bvis_name is None:
            bvis_name = 'bvis'
        if bhid_name is None:
            bhid_name = 'bhid'        
        
        if W is None:
            initial_W = np.asarray(np.random.normal(loc=0.01, scale=0.01, size=(n_visible,n_hidden)), dtype = config.floatX)
            W = shared(value=initial_W, name=W_name)   
            
        if bvis is None:
            bvis = shared(value=np.ones(n_visible, dtype = config.floatX), name = bvis_name)
                        
        if bhid is None:
            bhid = shared(value=np.ones(n_hidden, dtype = config.floatX), name = bhid_name)
                        
        
        super(ReluAutoEncoder,self).__init__(numpy_rng, theano_rng, input, n_visible, n_hidden, W, bhid, bvis, W_name, bvis_name, bhid_name)
        self.output = T.maximum(T.dot(input, self.W) + self.b, 0.0)
        
    @classmethod
    def class_from_values(cls, *args, **kwargs):
        """ This constructor is intended for dynamically constructing a dA layer subclass 
            Args that always get specified through this version of the constructor: 
                numpy_rng, theano_rng, input, n_visible, n_hidden, W_name, bvis_name, bhid_name.
            Args that *might* be specified: W, bhid, bvis. 
        """
        keys = kwargs.keys()
        if 'W' not in keys:
            kwargs['W'] = None
            
        if 'bhid' not in keys:
            kwargs['bhid'] = None
            
        if 'bvis' not in keys:
            kwargs['bvis']=None
            
            
        return cls(numpy_rng=kwargs['numpy_rng'], theano_rng=kwargs['theano_rng'], input=kwargs['input'], n_visible=kwargs['n_visible'],
                   n_hidden=kwargs['n_hidden'], W=kwargs['W'],
                   bhid=kwargs['bhid'], bvis=kwargs['bvis'],
                   W_name=kwargs['W_name'], bvis_name=kwargs['bvis_name'], bhid_name=kwargs['bhid_name'])        

    def get_reconstructed_input(self, hidden):
        """ Use a linear decoder to compute the reconstructed input given the hidden rep'n """
        return T.dot(hidden, self.W_prime) + self.b_prime
    
    def get_hidden_values(self,input):
        """ Apply ReLu elementwise to the transformed input """
        return T.maximum(T.dot(input, self.W) + self.b, 0.0)
    
    def get_cost_updates(self, corruption_level, learning_rate):
        """ Compute the reconstruction error over the mini-batched input
       taking into account a certain level of corruption of the input """
        x_corrupted = super(ReluAutoEncoder,self).get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(x_corrupted)
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
    
    def get_cost_gparams(self, corruption_level, learning_rate):
        """ Compute the reconstruction error over the mini-batched input (with corruption)
    
        But instead of returning a list of tuples (updates) were the rval has the form of an update to 
        a theano.tensor variable (param, update_value), return instead (param, gparam)."""
        x_corrupted = super(ReluAutoEncoder,self).get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(x_corrupted)
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
            updates.append((param, learning_rate * gparam))
            
        return (cost, updates)    