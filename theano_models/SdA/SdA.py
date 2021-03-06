import numpy as np  

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import shared

from collections import OrderedDict
from logistic_sgd import LogisticRegression
from AutoEncoder import AutoEncoder, BernoulliAutoEncoder, GaussianAutoEncoder, ReluAutoEncoder


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
                 loss='squared', dropout_rates = None, sparse_init=-1, opt_method = 'NAG'):

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
                            
        :type sparse_init: int
        :param sparse_init: Initialize the weight matrices using Martens sparse initialization (Martens ICML 2010)
                            >0 specifies the number of units in the layer that have initial weights drawn from 
                            a N(0,1).  Use -1 for dense init.

        :type opt_method: string
        :param opt_method: specifies the optimization method used to fit the model parameters.  
                            Accepted values are {'CM': Classical Momentum, 'NAG': Nesterov Accelerated Gradient.}                                                                                            
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
        
        # sanity check on optimization method 
        assert opt_method.upper() in ['CM','NAG']
        self.opt_method = opt_method.upper()
        
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
                            bhid_name=bhid_name,
                            sparse_init=sparse_init)         
                
            self.dA_layers.append(dA_layer)
            self.params.extend(dA_layer.params)            
            

        # Keep track of parameter updates so we may use momentum 
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
            self.finish_sda_unsupervised()

    def finish_sda_unsupervised(self):    
        """ Finish up unsupervised property settings for the model: set self.loss, self.finetune_cost, self.output, self.errors """
        
        loss_dict = {'squared': self.squared_loss, 'xent': self.xent_loss, 'softplus': self.softplus_loss}
        self.loss = loss_dict[self.use_loss]
        self.finetune_cost = self.reconstruction_error(self.x)
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
            
                
    def reconstruct_input(self, X):
        """ Given data X, provide the symbolic computation of  
        \hat{X} where \hat{X} is the reconstructed data output of the 'unrolled' SdA
         
        
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
    
    def reconstruct_input_limited(self, X, i):
        """ Given data X, provide the symbolic computation of 
        \hat{X} where \hat{X} is the reconstructed data output 
        using only the first i (counting from 0) layers of the 'unrolled' SdA """
        
        X_prime = X
        for dA in self.dA_layers[:i]:
            X_prime = dA.get_hidden_values(X_prime)
        
        for dA in self.dA_layers[i-1::-1]:
            X_prime = dA.get_reconstructed_input(X_prime)
        return X_prime    
    
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
    
    def reconstruction_error_limited(self, X, limit):
        """ Calculate the reconstruction error using a limited number of layers
        in the SdA.
        
        :type X: theano.tensor.TensorType
        :param X: Shared variable that contains a batch of datapoints 
                  to be reconstructed
                  
        :type limit: int
        :param limit: Use the first 'limit' layers of the SdA for reconstruction 
        """
        
        Z = self.reconstruct_input_limited(X, limit)    
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
    
##############################  Regularization functions  #########

    def max_norm_regularization(self):
        '''
        Define and return a list of theano function objects implementing max norm 
        regularization for each weight matrix in each layer of the SdA.  
        
        '''
        
        norm_limit = T.scalar('norm_limit')
        max_norm_updates = OrderedDict()
        
        for param in self.params:            
            if param.get_value(borrow=True).ndim == 2:
                # max-norm column regularization as per Pylearn2 MLP lib
                col_norms = T.sqrt(T.sum(T.sqr(param), axis=0))
                desired_norms = T.clip(col_norms, 0, norm_limit)
                updated_W = param * (desired_norms / (1e-7 + col_norms))  
                max_norm_updates[param] = updated_W
        fn = theano.function([norm_limit], [], updates = max_norm_updates)
        return fn

    def nag_param_update(self):
        ''' Define and return a theano function to apply momentum updates to each 
        parameter that is part of momentum updates '''
        
        momentum = T.fscalar('momentum')
        delta_t_updates = OrderedDict()
        for param in self.params:
            if param in self.updates:
                delta_t = self.updates[param]              
                delta_t_updates[param] = param + momentum * delta_t
        fn = theano.function([momentum], [], updates = delta_t_updates)        
        return fn    
        

    def sgd_cm(self, learning_rate, momentum, gparams):
        ''' Returns a dictionary of theano symbolic variables indicating how
        the shared variable parameters in the SdA should be updated, using classical 
        momentum. 
            
        N.B: learning_rate should be a theano.shared variable declared in the
        code driving the (pre)training of this SdA.
    
        :type momentum: theano.TensorVariable
        :param momenum: momentum parameter for SGD parameter updates
        
        :type learning_rate: theano.tensor.shared
        :param learning_rate: the learning rate for pretraining   
        
        :type gparams: list of tuples
        :param gparams: list of tuples, each of which contains (param, gparam) 
        i.e the partial derivative of cost by each SdA parameter '''       
        
        updates = OrderedDict()
        for param, grad_update in gparams:
            if param in self.updates:
                last_update = self.updates[param]
                delta = momentum * last_update - learning_rate * grad_update
                updates[param] = param + delta
                # update value of theano.shared in self.updates[param]
                updates[last_update] = delta
        return updates
    
    def sgd_cm_wd(self, learning_rate, momentum, weight_decay, gparams):
            ''' Returns a dictionary of theano symbolic variables indicating how
            the shared variable parameters in the SdA should be updated, using classical 
            momentum. 
                
            N.B: learning_rate should be a theano.shared variable declared in the
            code driving the (pre)training of this SdA.
        
            :type momentum: theano.TensorVariable
            :param momenum: momentum parameter for SGD parameter updates
            
            :type weight_decay: theano.TensorVariable
            :param weight_decay: weight decay regularization parameter for SGD parameter updates
            
            :type learning_rate: theano.tensor.shared
            :param learning_rate: the learning rate for pretraining   
            
            :type gparams: list of tuples
            :param gparams: list of tuples, each of which contains (param, gparam) 
            i.e the partial derivative of cost by each SdA parameter '''       
            
            updates = OrderedDict()
            for param, grad_update in gparams:
                if param in self.updates:
                    last_update = self.updates[param]
                    delta = momentum * last_update - learning_rate * grad_update - learning_rate * weight_decay * last_update
                    updates[param] = param + delta
                    # update value of theano.shared in self.updates[param]
                    updates[last_update] = delta
            return updates    
    
    def sgd_adagrad_momentum(self, momentum, learning_rate, gparams):
        ''' Returns a dictionary of theano symbolic variables indicating how
        the shared variable parameters in the SdA should be updated, using AdaGrad
        but with a decaying average of the gradients rather than sum
    
        :type momentum: theano.TensorVariable
        :param momenum: momentum parameter for SGD parameter updates
        
        :type learning_rate: theano.tensor.shared
        :param learning_rate: the base or master learning rate shared for all parameters
        
        :type gparams: list of tuples
        :param gparams: list of tuples, each of which contains (param, gparam) 
        i.e the partial derivative of cost by each SdA parameter   '''
        
        updates = OrderedDict()
        for param, gparam in gparams:
            grad_sqrd_hist = self.updates[param]
            grad_sqrd = momentum * grad_sqrd_hist + (1 - momentum) * (gparam **2) 
            param_update_val = param - learning_rate * gparam / (1e-7 + (grad_sqrd)** 0.5)
            updates[param] = param_update_val
            # update value of theano.shared in self.updates[param]
            updates[grad_sqrd_hist] = grad_sqrd
                
        return updates
    
    def sgd_adagrad_momentum_wd(self, momentum, learning_rate, weight_decay, gparams):
            ''' Returns a dictionary of theano symbolic variables indicating how
            the shared variable parameters in the SdA should be updated, using AdaGrad
            but with a decaying average of the gradients rather than sum
        
            :type momentum: theano.TensorVariable
            :param momenum: momentum parameter for SGD parameter updates
            
            :type weight_decay: theano.TensorVariable
            :param weight_decay: weight decay regularization parameter for SGD parameter updates
            
            :type learning_rate: theano.tensor.shared
            :param learning_rate: the base or master learning rate shared for all parameters
            
            :type gparams: list of tuples
            :param gparams: list of tuples, each of which contains (param, gparam) 
            i.e the partial derivative of cost by each SdA parameter   '''
            
            updates = OrderedDict()
            for param, gparam in gparams:
                grad_sqrd_hist = self.updates[param]
                grad_sqrd = momentum * grad_sqrd_hist + (1 - momentum) * (gparam **2) 
                param_update_val = param - learning_rate * gparam / (1e-7 + (grad_sqrd)** 0.5) - learning_rate * weight_decay * param
                updates[param] = param_update_val
                # update value of theano.shared in self.updates[param]
                updates[grad_sqrd_hist] = grad_sqrd
                    
            return updates    
    
    def sgd_adagrad(self, learning_rate, gparams):
        ''' Returns a dictionary of theano symbolic variables indicating how
        the shared variable parameters in the SdA should be updated, using AdaGrad
        
        :type learning_rate: theano.tensor.shared
        :param learning_rate: the base or master learning rate shared for all parameters
        
        :type gparams: list of tuples
        :param gparams: list of tuples, each of which contains (param, gparam) 
        i.e the partial derivative of cost by each SdA parameter   '''  
        
        updates = OrderedDict()
        for param, gparam in gparams:
            grad_sqrd_hist = self.updates[param]
            grad_sqrd = grad_sqrd_hist + gparam **2
            param_update_val = param - learning_rate * gparam / (1e-7 + (grad_sqrd)** 0.5)
            updates[param] = param_update_val
            updates[grad_sqrd_hist] = grad_sqrd
        
        return updates
                
    
    
##############################  Training functions ##########################


    def pretraining_functions(self, train_set_x, batch_size, learning_rate,method='cm'):
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
        
        :type method: string
        :param method: specifies the flavour of SGD used to train each dA layer.  Accepted values are 'cm', 'adagrad', 'adagrad_momentum'
        '''

        # index to a minibatch
        index = T.lscalar('index') 
        # % of corruption to use
        corruption_level = T.scalar('corruption')
        # momentum rate to use
        momentum = T.scalar('momentum')  
        
        assert method in ['cm','adagrad','adagrad_momentum']
        
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_gparams(corruption_level,learning_rate)
            
            # apply the updates in accordnace with the SGD method
            if method == 'cm':
                mod_updates = self.sgd_cm(learning_rate, momentum, updates)
                input_list = [index,momentum,theano.Param(corruption_level, default=0.25)]
            elif method == 'adagrad':
                mod_updates = self.sgd_adagrad(learning_rate, updates)
                input_list = [index,theano.Param(corruption_level, default=0.25)]
            else:
                mod_updates = self.sgd_adagrad_momentum(momentum, learning_rate, updates)
                input_list = [index,momentum,theano.Param(corruption_level, default=0.25)]
                
            # compile the theano function
            fn = theano.function(inputs=input_list, 
                                 outputs=cost,
                                 updates=mod_updates,
                                 givens={self.x: train_set_x[batch_begin:
                                                             batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)
            
        return pretrain_fns

    
    def build_finetune_limited_reconstruction(self, train_set_x, batch_size, learning_rate, method='cm'):
        ''' Generates a list of theano functions, each of them implementing one
        step in hybrid pretraining.  Hybrid pretraining is traning to minimize the 
        reconstruction error of the data against the representation produced using 
        two or more layers of the SdA.  
        
        N.B: learning_rate should be a theano.shared variable declared in the
        code driving the (pre)training of this SdA.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch
        
        :type learning_rate: theano.tensor.shared
        :param learning_rate: the learning rate for pretraining 
        
        :type method: string
        :param method: specifies the flavour of SGD used to train each dA layer.  Accepted values are 'cm', 'adagrad', 'adagrad_momentum' '''
        
        # index to a minibatch
        index = T.lscalar('index') 

        # momentum rate to use
        momentum = T.scalar('momentum')
        
        # weight decay to use
        weight_decay = T.scalar('weight_decay')
        
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size      
        
        # sanity check on number of layers
        assert 2 < len(self.dA_layers)
        
        # Check on SGD method
        assert method in ['cm','adagrad','adagrad_momentum','cm_wd','adagrad_momentum_wd']
        
        hybrid_train_fns = []
        for i in xrange(2,len(self.dA_layers)):

            # get the subset of model params involved in the limited reconstruction
            limited_params = self.params[:i*3]
                
            # compute the gradients with respect to the partial model parameters
            gparams = T.grad(self.reconstruction_error_limited(self.x, i), limited_params)
            
            # Ensure that gparams has same size as limited_params
            assert len(gparams) == len(limited_params)
            
            
            # apply the updates in accordnace with the SGD method
            if method == 'cm':
                mod_updates = self.sgd_cm(learning_rate, momentum, zip(limited_params,gparams))
                input_list = [index,momentum]
            elif method == 'adagrad':
                mod_updates = self.sgd_adagrad(learning_rate, zip(limited_params,gparams))
                input_list = [index]
            elif method == 'adagrad_momentum':
                mod_updates = self.sgd_adagrad_momentum(momentum, learning_rate, zip(limited_params,gparams))
                input_list = [index,momentum]
            elif method == 'cm_wd':
                mod_updates = self.sgd_cm_wd(learning_rate, momentum, weight_decay, zip(limited_params,gparams))
                input_list = [index,momentum,weight_decay]
            else:
                mod_updates = self.sgd_adagrad_momentum_wd(momentum, learning_rate, weight_decay, zip(limited_params,gparams))
                input_list = [index,momentum,weight_decay]            
                
            # the hybrid pre-training function now takes into account the update algorithm and proper input
            fn = theano.function(inputs=input_list, 
                                 outputs=self.reconstruction_error_limited(self.x, i),
                                 updates=mod_updates,
                                 givens={self.x: train_set_x[batch_begin:
                                                             batch_end]})
            # append `fn` to the list of functions
            hybrid_train_fns.append(fn)
            
        return hybrid_train_fns
    
    def build_finetune_full_reconstruction(self, datasets, batch_size, learning_rate, method='cm'):
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
        
        :type method: string
        :param method: specifies the flavour of SGD used to train each dA layer.  Accepted values are 'cm', 'adagrad', 'adagrad_momentum'
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
        
        # weight decay value to use
        weight_decay = T.scalar('weight_decay')
        
        assert method in ['cm','adagrad','adagrad_momentum','cm_wd','adagrad_momentum_wd']

        # apply the updates in accordnace with the SGD method
        if method == 'cm':
            mod_updates = self.sgd_cm(learning_rate, momentum, zip(self.params,gparams))
            input_list = [index,momentum]
        elif method == 'adagrad':
            mod_updates = self.sgd_adagrad(learning_rate, zip(self.params,gparams))
            input_list = [index]
        elif method == 'adagrad_momentum':
            mod_updates = self.sgd_adagrad_momentum(momentum, learning_rate, zip(self.params,gparams))
            input_list = [index,momentum]
        elif method == 'cm_wd':
            mod_updates = self.sgd_cm_wd(learning_rate, momentum, weight_decay, zip(self.params,gparams))
            input_list = [index,momentum,weight_decay]
        else:
            mod_updates = self.sgd_adagrad_momentum_wd(momentum, learning_rate, weight_decay, zip(self.params,gparams))
            input_list = [index,momentum,weight_decay]
                
        # compile the fine-tuning theano function, taking into account the update algorithm
        train_fn = theano.function(inputs=input_list,
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
        
        return (self.n_layers, self.n_outs, W_list, bhid_list, bvis_list, self.corruption_levels, self.layer_types, self.use_loss, self.dropout_rates, self.opt_method)
    
    def __setstate__(self, state):
        """ Unpickle an SdA model by restoring the list of dA layers.  
        The input should be provided to the initial layer, and the input of layer i+1 is set to the output of layer i. 
        Fill up the self.params from the dA params lists. """
        
        (layers, n_outs, dA_W_list, dA_bhid_list, dA_bvis_list, corruption_levels, layer_types, use_loss, dropout_rates, opt_method) = state
        self.n_layers = layers
        self.n_outs = n_outs
        self.corruption_levels = corruption_levels
        self.layer_types = layer_types
        self.dA_layers = []
        self.use_loss = use_loss
        self.opt_method = opt_method
        self.params = []
        self.x = T.matrix('x')  # symbolic input for the training data
        self.x_prime = T.matrix('X_prime') # symbolic output for the top layer dA
        
        numpy_rng = np.random.RandomState(123)
        theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))    
        
        # Set the dropout rates
        if dropout_rates is not None:
            self.dropout_rates = dropout_rates
        else:
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
                