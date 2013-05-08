import theano.tensor as T
from hidden_layer  import HiddenLayer
from logistic_sgd import LogisticRegression

class MLP(object):

	def __init__(self, rng, input, n_in, n_hidden, n_out):

		# hidden layer, defined in HiddenLayer.py
		self.hiddenLayer = HiddenLayer(rng = rng, input = input,
		n_in = n_in, n_out = n_hidden, activation = T.tanh)
	
		# output layer, logistic regression
		self.logRegressionLayer = LogisticRegression(input = self.hiddenLayer.output, 
		n_in = n_hidden, n_out = n_out)

		# Regularization of params
		# option 1: L1 regularization of params
		self.L1 = abs(self.hiddenLayer.W).sum() \
			+ abs(self.logRegressionLayer.W).sum()

		self.L2_sqr = (self.hiddenLayer.W **2).sum() \
				+ (self.logRegressionLayer.W **2).sum()

		# Define the log likelihood, errors based on component models

		self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood 

		self.errors = self.logRegressionLayer.errors

		self.params = self.hiddenLayer.params + \
				self.logRegressionLayer.params
		
	def __getstate__(self):
		""" Return the hidden layer and logistic regression layer that make up the MLP. """
		return (self.hiddenLayer,self.logRegressionLayer)
	
	def __setstate__(self, state):
		""" Re-establish the hidden layer and logistic regression layer objects. """
		(hiddenLayer,logRegressionLayer) = state
		self.hiddenLayer = hiddenLayer
		self.logRegressionLayer = logRegressionLayer
	
	def reconstruct_state(self,input, activation):
		""" Re-establish the inputs for each layer of the MLP. """
		self.hiddenLayer.reconstruct_state(input,activation)
		self.logRegressionLayer.reconstruct_state(self.hiddenLayer.output)
		self.L1 = abs(self.hiddenLayer.W).sum() \
				        + abs(self.logRegressionLayer.W).sum()
		
		self.L2_sqr = (self.hiddenLayer.W **2).sum() \
	                        + (self.logRegressionLayer.W **2).sum()

		# Define the log likelihood, errors based on component models

		self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood 

		self.errors = self.logRegressionLayer.errors

		self.params = self.hiddenLayer.params + \
	                        self.logRegressionLayer.params		
		
