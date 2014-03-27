import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import shared

from mlp.logistic_sgd import LogisticRegression
from dA.AutoEncoder import AutoEncoder, BernoulliAutoEncoder, GaussianAutoEncoder, ReluAutoEncoder

class SdA(object):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=-1,
                 corruption_levels=[0.1, 0.1], layer_types=['ReLU','ReLU']):
        """ This class is made to support a variable number of layers

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                          weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network.  Negative if 
                       there is no logistic layer on top.
        
        :type log_top: boolean
        :param log_top: True if a logistic regression layer should be stacked
                        on top of all the other layers

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
                                  
        :type layer_types: list of string
        :param layer_types: each entry specifies the AutoEncoder sub-class to
                            instatiate for each layer.
                                                                       
        """

        self.dA_layers = []
        self.params = []
        
        # Keep track of previous parameter updates so we can use momentum
        self.updates = {}
        
        self.n_outs = n_outs
        self.corruption_levels = corruption_levels
        self.n_layers = len(hidden_layers_sizes)

        # Sanity checks on parameter list sizes
        assert self.n_layers > 0
        assert len(hidden_layers_sizes) == len(corruption_levels) == len(layer_types)                                                                        

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the training input
        self.x_prime = T.matrix('X_prime') # the encoded output of the highest layer
        
        if n_outs > 0:
            self.y = T.ivector('y')  # the labels (if present) are presented as 1D vector of
                                     # [int] labels

        
        # Build each layer dynamically 
        layer_classes = {'Gaussian': GaussianAutoEncoder, 'Bernoulli': BernoulliAutoEncoder, 'ReLU': ReluAutoEncoder}
        
        for i in xrange(self.n_layers):
            
            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer.  
            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer            
            if i == 0:
                input_size = n_ins
                layer_input = self.x
            else:
                input_size = hidden_layers_sizes[i - 1]
                layer_input = self.dA_layers[-1].output

            # Call the appropriate dA subclass constructor
            dA_layer = layer_classes[layer_types[i]].class_from_values(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=int(hidden_layers_sizes[i]))         
                
            self.dA_layers.append(dA_layer)
            self.params.extend(dA_layer.params)            
            

        # Keep track of parameter updates, so we may use momentum 
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            update_name = param.name + '_update'
            self.updates[param] = theano.shared(init, name=update_name)        
            

        if n_outs > 0:
            self.logLayer = LogisticRegression(
                             input=self.dA_layers[-1].output,
                             n_in=hidden_layers_sizes[-1], n_out=n_outs)
    
            self.params.extend(self.logLayer.params)
        
            # compute the cost for second phase of training,
            # defined as the negative log likelihood
            self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
            
            # compute the gradients with respect to the model parameters
            # symbolic variable that points to the number of errors made on the
            # minibatch given by self.x and self.y
            self.errors = self.logLayer.errors(self.y)
        
        else:
            self.finetune_cost = self.reconstruction_error(self.x)
            self.output = self.encode(self.x)
            self.errors = self.reconstruction_error(self.x)
                    
            
    def reconstruct_input(self, X):
        """ Given data X, provide the symbolic computation of  
        \hat{X} where \hat{X} is the reconstructed data vector output of the 'unrolled' SdA 
        
        :type X: theano.tensor.TensorType
        :param X: Shared variable that contains data 
                  to be pushed through the SdA (i.e reconstructed)
        """
        X_prime = X
        for dA in self.dA_layers:
            X_prime = dA.get_hidden_values(X_prime)
        
        for dA in self.dA_layers[::-1]:
            X_prime = dA.get_reconstructed_input(X_prime)
        return X_prime
             
    
    def reconstruction_error(self, X):
        """ Calculate the reconstruction error. Take a matrix of 
        training examples where X[i,:] is one data vector, return 
        the squared error between X, Z where Z is the reconstructed data. 
        
        :type X: theano.tensor.TensorType
        :param X: Shared variable that contains a batch of datapoints 
                  to be reconstructed
        """
        Z = self.reconstruct_input(X)
        L = T.sum((X - Z) **2, axis = 1)
        return T.mean(L)
            
    def encode(self,X):
        """ Given data X, provide the symbolic computation of X_prime, by 
        passing X forward through to the top (lowest dimensional) layer of 
        the SdA 
        
        :type X: theano.tensor.TensorType
        :param X: Shared variable that contains data 
                  to be pushed through the SdA (i.e reconstructed)
        """
        X_prime = X
        for dA in self.dA_layers:
            X_prime = dA.get_hidden_values(X_prime) 
        
        self.x_prime = X_prime
        return self.x_prime    

    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in training the dA corresponding to the layer with same index.
        The function takes a minibatch index, and so training one dA layer
        corresponds to iterating this layer-specific training function in the
        list over all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        '''

        # index to a minibatch
        index = T.lscalar('index') 
        # % of corruption to use
        corruption_level = T.scalar('corruption')
        # learning rate to use
        learning_rate = T.scalar('lr')
        # momentum rate to use
        momentum = T.scalar('momentum')
        # weight decay to use
        weight_decay = T.scalar('weight_decay')  
        
        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,learning_rate)
            
            # modify the updates to account for the momentum smoothing and weight decay regularization
            
            mod_updates = []
            for param, grad_update in updates:
                if param in self.updates:
                    last_update = self.updates[param]
                    delta = momentum * last_update - weight_decay * learning_rate * param - learning_rate * grad_update
                    mod_updates.append((param, param + delta))
                    mod_updates.append((last_update, delta))
                else:               
                    mod_updates.append((param, grad_update))
            
                
            # compile the theano function
            fn = theano.function(inputs=[index,
                              theano.Param(corruption_level, default=0.15),
                              theano.Param(learning_rate, default=0.001),
                              theano.Param(momentum, default=0.8),
                              theano.Param(weight_decay, default=0.)],
                                 outputs=cost,
                                 updates=mod_updates,
                                 givens={self.x: train_set_x[batch_begin:
                                                             batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)
            
        return pretrain_fns


    def build_finetune_functions_reconstruction(self, datasets, batch_size, learning_rate):
        ''' 
        Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the reconstruction 
        error on a batch from the validation set

        :type datasets: tuple of theano.tensor.TensorType
        :param datasets: A tuple of two datasets;
                         `train`, `valid` in this order, each 
                         one is a T.dmatrix of datapoints

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''
        (train_set_x, valid_set_x) = datasets
        
        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        
        #DEBUG
        print "...number of validation batches: " + str(n_valid_batches)
        
        index = T.lscalar('index')  # index to a [mini]batch     
        
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)       

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))
                    

        train_fn = theano.function(inputs=[index],
              outputs=self.finetune_cost,
              updates=updates,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size]}, mode='DebugMode')

        valid_score_i = theano.function([index], self.errors,
              givens={
                 self.x: valid_set_x[index * batch_size:
                                     (index + 1) * batch_size]})

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        return train_fn, valid_score        
    
    def build_encoding_functions(self, dataset):
        ''' Generates a function `encode` that feeds the data forward 
        through the layers of the SdA and results in a lower dimensional
        output, which is the representation of the highest layer.

        :type dataset: theano.tensor.TensorType
        :param dataset: A T.dmatrix of datapoints to be fed through the SdA
                         
        '''

        start = T.lscalar('start')
        end = T.lscalar('end')

        encode_fn = theano.function(inputs=[start,end],
              outputs=self.output,
              givens={self.x: dataset[start:end]})
        return encode_fn
    
    
    def __getstate__(self):
        """ Pickle this SdA by returning the number of layers, list of sigmoid layers and list of dA layers. """
        return (self.n_layers, self.n_outs, self.dA_layers, self.corruption_levels)
    
    def __setstate__(self, state):
        """ Unpickle an SdA model by restoring the list of dA layers.  
        The input should be provided to the initial layer, and the input of layer i+1 is set to the output of layer i. 
        Fill up the self.params list with the parameter sets of the dA list. """
        
        (layers, n_outs, dA_layers_list, corruption_levels) = state
        self.n_layers = layers
        self.n_outs = n_outs
        self.corruption_levels = corruption_levels
        self.dA_layers = []
        self.params = []
        self.x = T.matrix('x')  # symbolic input for the training data
        self.x_prime = T.matrix('X_prime') # symbolic output for the top layer dA
        
        numpy_rng = np.random.RandomState(123)
        theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))        
        
        # N.B: keys for this dict differ than those in the constructor, since the strings 
        # returned by dA_layers_list[i].__class__.__name__ are class names.
        layer_classes = {'GaussianAutoEncoder': GaussianAutoEncoder, 'BernoulliAutoEncoder': BernoulliAutoEncoder, 'ReluAutoEncoder': ReluAutoEncoder}
           
        for i in xrange(self.n_layers):
            
            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.dA_layers[i-1].output
            
            layer_type = dA_layers_list[i].__class__.__name__
            
                # Rebuild the dA layer from scratch, explicitly tying the W,bhid params to those from the sigmoid layer
            dA_layer = layer_classes[layer_type](numpy_rng=numpy_rng,
                        theano_rng=theano_rng,
                        input=layer_input,
                        n_visible=dA_layers_list[i].n_visible,
                        n_hidden=dA_layers_list[i].n_hidden,
                        W=dA_layers_list[i].W,
                        bhid=dA_layers_list[i].b,
                        bvis=dA_layers_list[i].b_prime) 
                
            self.dA_layers.append(dA_layer)
            self.params.extend(self.dA_layers[i].params)
            
        # Reconstruct the dictionary of shared vars for parameter updates 
        # so we can use momentum when training.
        self.updates = {}
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            update_name = param.name + '_update'
            self.updates[param] = theano.shared(init, name=update_name)
            
        # Reconstruct the finetuning cost functions
        if n_outs > 0:
            self.reconstruct_loglayer(n_outs)
        else:
            self.finetune_cost = self.reconstruction_error(self.x)
            self.errors = self.reconstruction_error(self.x)
            self.output = self.encode(self.x)

            
    def reconstruct_loglayer(self, n_outs = 10):
        """ Reconstruct a logistic layer on top of a previously trained SdA """
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
                         input=self.dA_layers[-1].output,
                         n_in=self.dA_layers[-1].n_hidden, n_out=n_outs)

        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)        

        
    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)       

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(inputs=[index],
              outputs=self.finetune_cost,
              updates=updates,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]})

        test_score_i = theano.function([index], self.errors,
                 givens={
                   self.x: test_set_x[index * batch_size:
                                      (index + 1) * batch_size],
                   self.y: test_set_y[index * batch_size:
                                      (index + 1) * batch_size]})

        valid_score_i = theano.function([index], self.errors,
              givens={
                 self.x: valid_set_x[index * batch_size:
                                     (index + 1) * batch_size],
                 self.y: valid_set_y[index * batch_size:
                                     (index + 1) * batch_size]})

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score                
                