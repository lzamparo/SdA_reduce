import pdb
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import shared

from mlp.logistic_sgd import LogisticRegression, load_data
from mlp.hidden_layer import HiddenLayer
from dA.AutoEncoder import AutoEncoder

class SdA(object):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10,
                 log_top=False, corruption_levels=[0.1, 0.1], 
                 dA_losses=['xent','xent']):
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
        :param n_outs: dimension of the output of the network 
        
        :type log_top: boolean
        :param log_top: True if a logistic regression layer should be stacked
        on top of all the other layers

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
                                  
        :type dA_loss: list of strings
        :param dA_loss: loss functions to use for each of the dA layers  
                                                            
        """

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        
        # Keep track of previous parameter updates so we can use momentum
        self.updates = {}
        
        self.n_outs = n_outs
        self.corruption_levels = corruption_levels
        self.n_layers = len(hidden_layers_sizes)

        # Sanity checks on parameter list sizes
        assert self.n_layers > 0
        assert len(hidden_layers_sizes) == len(dA_losses) 
        assert len(dA_losses) == len(corruption_levels)

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the training input
        self.y = T.ivector('y')  # the labels (if present) are presented as 1D vector of
                                 # [int] labels

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with different denoising autoencoders.
        #
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer.
        #
        # During pre-training we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well).
        #
        # During fine-tunining we will finish training the SdA by doing
        # stochastic gradient descent on the MLP

        for i in xrange(self.n_layers):
            
            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=int(hidden_layers_sizes[i]),
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = AutoEncoder(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=int(hidden_layers_sizes[i]),
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b,
                          loss=dA_losses[i])
            self.dA_layers.append(dA_layer)

        # keep track of parameter updates for pretraining
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            update_name = param.name + '_update'
            self.updates[param] = theano.shared(init, name=update_name)        
            

        # Add a logistic layer on top of the MLP ?
        if log_top:
            self.logLayer = LogisticRegression(
                             input=self.sigmoid_layers[-1].output,
                             n_in=hidden_layers_sizes[-1], n_out=n_outs)
    
            self.params.extend(self.logLayer.params)
        
                
        
            # construct a function that implements one step of finetunining
            # compute the cost for second phase of training,
            # defined as the negative log likelihood
            self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
            
            # compute the gradients with respect to the model parameters
            # symbolic variable that points to the number of errors made on the
            # minibatch given by self.x and self.y
            self.errors = self.logLayer.errors(self.y)

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
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            
            # modify the updates to account for the momentum smoothing and weight decay regularization
            # As returned from dA.get_cost_updates, the list of tuples goes like
            # param: (shared var) parameter , update: param - learning_rate * gparam
            
            #Can you check what the dtype of grad_update is when param == W?
            #It may be possible that a float64 gradient is returned for a
            #float32 parameter. If that is the case, you can cast it by using
            #"grad_update.astype(config.floatX)" for instance (assuming floatX ==
            #'float32').
            
            # Roll call!  everyone below print out your name and type
            mod_updates = []
            pdb.set_trace()
            for param, grad_update in updates:
                if param in self.updates:
                    last_update = self.updates[param]
                    print "Name: " + last_update.name + " type: " + str(last_update.type)
                    print "Name: " + param.name + " type: " + str(param.type)
                    print "Grade update for " + param.name  + " has type: " + str(grad_update.type)
                    delta = momentum * last_update - weight_decay * learning_rate * param - learning_rate * grad_update
                    mod_updates.append((param, param + delta))
                    mod_updates.append((last_update, delta))
                else:
                    print "Name: " + param.name + " type: " + str(param.type)
                    print "Grade update for " + param.name  + " has type: " + str(grad_update.type)                    
                    mod_updates.append((param, grad_update))
                
            pdb.set_trace()
            # compile the theano function
            fn = theano.function(inputs=[index,
                              theano.Param(corruption_level, default=0.2),
                              theano.Param(learning_rate, default=0.1),
                              theano.Param(momentum, default=0.),
                              theano.Param(weight_decay, default=0.)],
                                 outputs=cost,
                                 updates=mod_updates,
                                 givens={self.x: train_set_x[batch_begin:
                                                             batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

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

    
    def __getstate__(self):
        """ Pickle this SdA by returning the number of layers, list of sigmoid layers and list of dA layers. """
        return (self.n_layers, self.n_outs, self.sigmoid_layers, self.dA_layers, self.corruption_levels)
    
    def __setstate__(self, state):
        """ Unpickle an SdA model by restoring the lists of both MLP hidden layers and dA layers.  
        Reconstruct both MLP and stacked dA aspects of an unpickled SdA model.  The input should be provided to 
        the initial layer, and the input of layer i+1 is set to the output of layer i. 
        Fill up the self.params list with the parameter sets of the MLP list. """
        (layers, n_outs, mlp_layers_list, dA_layers_list, corruption_levels) = state
        self.n_layers = layers
        self.n_outs = n_outs
        self.corruption_levels = corruption_levels
        self.dA_layers = []
        self.params = []
        self.sigmoid_layers = mlp_layers_list
        self.x = T.matrix('x')  # symbolic input for the training data
        
        numpy_rng = np.random.RandomState(123)
        theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))        
           
        for i in xrange(self.n_layers):
            
            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[i-1].output
                
            self.sigmoid_layers[i].reconstruct_state(layer_input)
            self.params.extend(self.sigmoid_layers[i].params)
            
            # Rebuild the dA layer from scratch, explicitly tying the W,bhid params to those from the sigmoid layer
            dA_layer = AutoEncoder(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=dA_layers_list[i].n_visible,
                          n_hidden=dA_layers_list[i].n_hidden,
                          W=self.sigmoid_layers[i].W,
                          bhid=self.sigmoid_layers[i].b,
                          bvis=dA_layers_list[i].b_prime,
                          loss=dA_layers_list[i].loss)
            self.dA_layers.append(dA_layer)
            
        # Reconstruct the dictionary of shared vars for parameter updates 
        # so we can use momentum when training.
        self.updates = {}
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            update_name = param.name + '_update'
            self.updates[param] = theano.shared(init, name=update_name)        

            
    def reconstruct_loglayer(self, n_outs = 10):
        """ Reconstruct a logistic layer on top of a previously trained SdA """
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
                         input=self.sigmoid_layers[-1].output,
                         n_in=self.sigmoid_layers[-1]._outsize, n_out=n_outs)

        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)        

        
            