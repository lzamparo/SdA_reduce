import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import shared

from collections import OrderedDict
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
                 corruption_levels=[0.1, 0.1], layer_types=['ReLU','ReLU'],
                 loss='squared', dropout_rates = None):
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
                            
        :type loss: string
        :param loss: specify what loss function to use for reconstruction error
                            Currently supported: 'squared','xent','softplus'
                            
        :type dropout_rates: list of float
        :param dropout_rates: proportion of output units to drop from this layer
                            Default is to retain all units in all layers
                                  
                                                                       
        """

        self.dA_layers = []
        self.params = []
        self.layer_types = layer_types
        
        # keep track of previous parameter updates so we can use momentum
        self.updates = OrderedDict()
        
        self.n_outs = n_outs
        self.corruption_levels = corruption_levels
        self.n_layers = len(hidden_layers_sizes)

        # Calculate dropout params (or set if provided)
        if dropout_rates is not None:
            self.dropout_rates = dropout_rates
            assert len(dropout_rates) == len(layer_types)
            assert dropout_rates[-1] == 1.0
        else:
            self.dropout_rates = [1.0 for l in layer_types]

        # sanity checks on parameter list sizes
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

        # sanity check on loss parameter
        assert loss.lower() in ['squared', 'xent', 'softplus']
        self.use_loss = loss.lower()
        
        # build each layer dynamically 
        layer_classes = {'gaussian': GaussianAutoEncoder, 'bernoulli': BernoulliAutoEncoder, 'relu': ReluAutoEncoder}
        
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
            w_name = 'W_' + str(i)
            bvis_name = 'bvis_' + str(i)
            bhid_name = 'bhid_' + str(i)
            dA_layer = layer_classes[layer_types[i]].class_from_values(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=int(hidden_layers_sizes[i]),
                            W_name=w_name,
                            bvis_name=bvis_name,
                            bhid_name=bhid_name)         
                
            self.dA_layers.append(dA_layer)
            self.params.extend(dA_layer.params)            
            

        # Keep track of parameter updates for weight matrices, so we may use momentum 
        for param in self.params:
            if param.get_value(borrow=True).ndim == 2:
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
            self.finish_sda_unsupervised()

    def finish_sda_unsupervised(self):    
        """ Finish up unsupervised property settings for the model: set self.loss, self.finetune_cost, self.output, self.errors """
        
        loss_dict = {'squared': self.squared_loss, 'xent': self.xent_loss, 'softplus': self.softplus_loss}
        self.loss = loss_dict[self.use_loss]
        self.finetune_cost = self.reconstruction_error_dropout(self.x)
        self.output = self.encode(self.x)
        self.errors = self.reconstruction_error(self.x)        
                      
    def squared_loss(self,X,Z):
        """ Return the theano expression for squared error loss
        
        :type X: theano.tensor.TensorType
        :param X: Shared variable that contains data 
                  
        :type Z: theano.tensor.TensorType
        :param Z: Shared variable that contains the reconstruction
                  of the data under the model)          
        """
        
        return T.sum((X - Z) **2, axis = 1)
    
    def softplus_loss(self,X,Z):
        """ Return the theano expression for softplus error loss 

        :type X: theano.tensor.TensorType
        :param X: Shared variable that contains data 
                  
        :type Z: theano.tensor.TensorType
        :param Z: Shared variable that contains the reconstruction
                  of the data under the model)
        """
        
        return T.sum((X - T.nnet.softplus(Z)) **2, axis = 1)
    
    def xent_loss(self,X,Z):
        """ Return the theano expression for cross entropy error loss 
        
        :type X: theano.tensor.TensorType
        :param X: Shared variable that contains data 
                  
        :type Z: theano.tensor.TensorType
        :param Z: Shared variable that contains the reconstruction
                  of the data under the model)
        """
        
        return -T.sum(X * T.log(Z) + (1 - X) * T.log(1 - Z), axis=1)            
            
    def reconstruct_input_dropout(self, X):
        """ Given data X, provide the symbolic computation of  
        \hat{X} where \hat{X} is the reconstructed data vector output of the 'unrolled' SdA
        
        Apply a dropout mask to the output of the previous layer
        
        :type X: theano.tensor.TensorType
        :param X: Shared variable that contains data 
                  to be pushed through the SdA (i.e reconstructed)
        """
       
        X_prime = X
        for dA, p in zip(self.dA_layers,self.dropout_rates):
            hidden = dA.get_hidden_values(X_prime)
            X_prime = dA.dropout_from_layer(hidden,p)
        
        for dA in self.dA_layers[::-1]:
            X_prime = dA.get_reconstructed_input(X_prime)
        return X_prime
   
    
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
        L = self.loss(X,Z)
        return T.mean(L)
    
    def reconstruction_error_dropout(self, X):
        """ Calculate the reconstruction error. Take a matrix of 
        training examples where X[i,:] is one data vector, return 
        the squared error between X, Z where Z is the reconstructed data. 
        
        :type X: theano.tensor.TensorType
        :param X: Shared variable that contains a batch of datapoints 
                  to be reconstructed
        """
        
        Z = self.reconstruct_input_dropout(X)    
        L = self.loss(X,Z)
        return T.mean(L)    
    
    def scale_dA_weights(self,factors):
        """ Scale each dA weight matrix by some factor.  Used primarily when encoding 
        data trained with an SdA where droput was used in finetuning. 
        
        :type factors: list of floats
        :param factors: scale the weight matrices by the factors in the list
        """
        for dA,p in zip(self.dA_layers,factors):
            W,meh,bah = dA.get_params()
            W.set_value(W.get_value(borrow=True) * p, borrow=True)
            
                        
            
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
    
    def max_norm_regularization(self):
        '''
        Define and return a list of theano function objects implementing max norm 
        regularization for each weight matrix in each layer of the SdA.  
        
        '''
        
        norm_limit = T.scalar('norm_limit')
        max_norm_fns = []
        
        for dA in self.dA_layers:            
            W,scrub,dub = dA.get_params()
            # max-norm column regularization as per Pylearn2 MLP lib
            col_norms = T.sqrt(T.sum(T.sqr(W), axis=0))
            desired_norms = T.clip(col_norms, 0, norm_limit)
            updated_W = W * (desired_norms / (1e-7 + col_norms))            
            fn = theano.function([norm_limit], desired_norms, updates = {W: updated_W})
            max_norm_fns.append(fn)
            
        return max_norm_fns

##############################  Training functions ##########################


    def pretraining_functions(self, train_set_x, batch_size, learning_rate):
        ''' Generates a list of functions, each of them implementing one
        step in training the dA corresponding to the layer with same index.
        The function takes a minibatch index, and so training one dA layer
        corresponds to iterating this layer-specific training function in the
        list over all minibatch indexes.
        
        N.B: learning_rate should be a theano.shared variable declared in the
        code driving the (pre)training of this SdA.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch
        
        :type learning_rate: theano.tensor.shared
        :param learning_rate: the learning rate for pretraining 
        

        '''

        # index to a minibatch
        index = T.lscalar('index') 
        # % of corruption to use
        corruption_level = T.scalar('corruption')
        # momentum rate to use
        momentum = T.scalar('momentum')  
        
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_gparams(corruption_level,learning_rate)
            
            # modify the updates to account for momentum smoothing 
            momentum_updates = OrderedDict()
            for param, grad_update in updates:
                if param in self.updates:
                    last_update = self.updates[param]
                    delta = momentum * last_update - (1. - momentum) * grad_update
                    momentum_updates[param] = param + delta
                    self.updates[param] = delta
                else:               
                    momentum_updates[param] = param - grad_update
            
                
            # compile the theano function
            fn = theano.function(inputs=[index,
                              theano.Param(corruption_level, default=0.15),
                              theano.Param(momentum, default=0.8)], 
                                 outputs=cost,
                                 updates=momentum_updates,
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

        :type learning_rate: theano.tensor.shared
        :param learning_rate: learning rate used during finetune stage
        '''
        (train_set_x, valid_set_x) = datasets
        
        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        
        index = T.lscalar('index')  # index to a [mini]batch     
        
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)       
        
        # momentum rate to use
        momentum = T.scalar('momentum')        

        # package up each param with it's gradient component * learning rate
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, gparam * learning_rate))
            
        
        # modify the updates to account for momentum smoothing
        mod_updates = OrderedDict()
        for param, grad_update in updates:
            if param in self.updates:
                last_update = self.updates[param]
                delta = momentum * last_update - (1.0 - momentum) * grad_update
                mod_updates[param] = param + delta
                self.updates[param] = delta
            else:               
                mod_updates[param] = param - grad_update        
                

        train_fn = theano.function(inputs=[index, theano.Param(momentum, default=0.8)],
              outputs=self.finetune_cost,
              updates=mod_updates,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size]})

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
    
    
    def test_gradient(self,dataset,index=1,batch_size=1):
        ''' Return a Theano function that will evaluate
        the gradient wrt some points sampled from the provided dataset) 
        
        Example provided by http://deeplearning.net/software/theano/tutorial/gradients.html#tutcomputinggrads
        x = T.dmatrix('x')
        s = T.sum(1 / (1 + T.exp(-x)))
        gs = T.grad(s, x)
        dlogistic = function([x], gs)
        dlogistic([[0, 1], [-1, -2]])
        
        :type dataset: theano.tensor.TensorType
        :param dataset: A T.dmatrix of datapoints, should be a shared variable.
        
        :type index: int
        :param index: identifies the start of the gradient test batch of data, a subset of dataset.
        
        :type batch_size: int
        :param batch_size: size of the test batch.
        
        '''
                        
        index_val = T.lscalar('gtestindex')  # index to a [mini]batch     
                
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params) 
        
        # create a function to evaluate the gradient on the batch at index
        eval_grad = theano.function(inputs=[index_val], outputs=gparams, givens= {self.x: dataset[index_val * batch_size: (index_val + 1) * batch_size]})    
        return eval_grad
     
     
##################### Pickling functions ###############################     
        
    def __getstate__(self):
        """ Pickle this SdA by tupling up the layers, output size, dA param lists, corruption levels and layer types. """
        W_list = []
        bhid_list = []
        bvis_list = []
        for layer in self.dA_layers:
            W, bhid, bvis  = layer.get_params()
            W_list.append(W.get_value(borrow=True))
            bhid_list.append(bhid.get_value(borrow=True))
            bvis_list.append(bvis.get_value(borrow=True))
        
        return (self.n_layers, self.n_outs, W_list, bhid_list, bvis_list, self.corruption_levels, self.layer_types, self.use_loss)
    
    def __setstate__(self, state):
        """ Unpickle an SdA model by restoring the list of dA layers.  
        The input should be provided to the initial layer, and the input of layer i+1 is set to the output of layer i. 
        Fill up the self.params from the dA params lists. """
        
        (layers, n_outs, dA_W_list, dA_bhid_list, dA_bvis_list, corruption_levels, layer_types, use_loss) = state
        self.n_layers = layers
        self.n_outs = n_outs
        self.corruption_levels = corruption_levels
        self.layer_types = layer_types
        self.dA_layers = []
        self.use_loss = use_loss
        self.params = []
        self.x = T.matrix('x')  # symbolic input for the training data
        self.x_prime = T.matrix('X_prime') # symbolic output for the top layer dA
        
        numpy_rng = np.random.RandomState(123)
        theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))    
        
        # Set the default dropout rates, can be updated later in a driver script
        self.dropout_rates = [1.0 for i in xrange(self.n_layers)]
        
        # build each layer dynamically 
        layer_classes = {'gaussian': GaussianAutoEncoder, 'bernoulli': BernoulliAutoEncoder, 'relu': ReluAutoEncoder}
           
        for i in xrange(self.n_layers):
            
            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.dA_layers[i-1].output
            
            # Rebuild the dA layer from the values provided in layer_types, dA_<param>_lists            
            
            n_visible,n_hidden = dA_W_list[i].shape
            w_name = 'W_' + str(i)
            bhid_name = 'bhid_' + str(i)
            bvis_name = 'bvis_' + str(i)
            
            lt = layer_types[i].lower()
            dA_layer = layer_classes[lt](numpy_rng=numpy_rng,
                        theano_rng=theano_rng,
                        input=layer_input,
                        n_visible=n_visible,
                        n_hidden=n_hidden,
                        W=shared(value=dA_W_list[i],name=w_name),
                        bhid=shared(value=dA_bhid_list[i],name=bhid_name),
                        bvis=shared(value=dA_bvis_list[i],name=bvis_name)) 
                
            self.dA_layers.append(dA_layer)
            self.params.extend(self.dA_layers[i].params)
            
        # Reconstruct the dictionary of shared vars for parameter updates 
        # so we can use momentum when training.
        # Apply only to weight matrices
        self.updates = {}
        for param in self.params:
            if param.get_value(borrow=True).ndim == 2:
                init = np.zeros(param.get_value(borrow=True).shape,
                                dtype=theano.config.floatX)
                update_name = param.name + '_update'
                self.updates[param] = theano.shared(init, name=update_name)
            
        # Reconstruct the finetuning cost functions
        if n_outs > 0:
            self.reconstruct_loglayer(n_outs)
        else:
            self.finish_sda_unsupervised()

   
#################### Legacy code below: logistic layer top for SdA that were intended for dual MLP
#################### and the associated supervised fine-tuning function.
        
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
                