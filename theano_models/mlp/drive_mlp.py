# try out the MLP on MNIST classification

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression, load_data
from hidden_layer import HiddenLayer
from MLP import MLP

def drive_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, 
	n_epochs=1000, dataset='../data/mnist.pkl.gz', batch_size=20, 
	n_hidden=500):

	datasets = load_data(dataset)

	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]

	# compute number of minibatches for each set
	n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
	n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

	### Build the model ###
	print '... building the model'

	# allocate symbolic variables for the data
	index = T.lscalar()
	x = T.matrix('x')
	y = T.ivector('y')

	rng = numpy.random.RandomState(1234)

	# construct the MLP class
	classifier = MLP(rng = rng, input = x, n_in=28*28, n_hidden=n_hidden, n_out=10)

	# cost to be minimized
	cost = classifier.negative_log_likelihood(y) \
		+ L1_reg * classifier.L1 \
		+ L2_reg * classifier.L2_sqr

	# theano function that computes the mistakes made by the model on a minibatch
	test_model = theano.function(inputs=[index],
		outputs = classifier.errors(y),
		givens={
			x: test_set_x[index * batch_size:(index + 1) * batch_size],
			y: test_set_y[index * batch_size:(index + 1) * batch_size]})

	# theano function to validate the model
	validate_model = theano.function(inputs=[index],
		outputs = classifier.errors(y),
		givens = {
			x: valid_set_x[index * batch_size:(index + 1) * batch_size],
			y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

	# compute the gradient of the cost function w.r.t theta
	gparams = []
	for param in classifier.params:
		gparam = T.grad(cost, param)
		gparams.append(gparam)

	# build the list of parameter updates.  This consists of tuples of paramters and values
	updates = []

	for param, gparam in zip(classifier.params, gparams):
		updates.append((param, param - learning_rate * gparam))

	# compile a Theano function to return the cost, update the parameters based on the 
	# updates list
	train_model = theano.function(inputs=[index], outputs=cost,
		updates=updates,
		givens={
			x: train_set_x[index * batch_size:(index + 1) * batch_size],
			y: train_set_y[index * batch_size:(index + 1) * batch_size]})

	### train the model ###
	print '... training'

	# early-stopping parameters
	patience = 10000 		# look at this number of examples regardless
	patience_increase = 2 	# wait this many more epochs when a new best comes up
	improvement_threshold = 0.995	# a relative improvement threshold for significance

	validation_frequency = min(n_train_batches, patience / 2) 
		# train for this many minibatches before checking the model on the validation set

	best_params = None
	best_validation_loss = numpy.inf
	best_iter = 0
	test_score = 0.
	start_time = time.clock()

	epoch = 0
	done_looping = False

	while (epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1
		for minibatch_index in xrange(n_train_batches):

			minibatch_avg_cost = train_model(minibatch_index)
			iter = (epoch - 1) * n_train_batches + minibatch_index
		
			# do we validate?
			if (iter + 1) % validation_frequency == 0:
				validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
				this_validation_loss = numpy.mean(validation_losses)

				print('epoch %i, minibatch %i/%i, validation error %f %%' % 
					(epoch, minibatch_index + 1, n_train_batches,
					 this_validation_loss * 100.))

				if this_validation_loss < best_validation_loss:
					# increase patience if loss improvement is good enough
					if this_validation_loss < best_validation_loss * \
							improvement_threshold:
						patience = max(patience, iter * patience_increase)

					best_validation_loss = this_validation_loss
					best_iter = iter
		
					# test on the test set
					test_losses = [test_model(i) for i in xrange(n_test_batches)]
					test_scores = numpy.mean(test_losses)

					print(('	epoch %i, minibatch %i/%i, test error of '
						'best model %f %%') %
						(epoch, minibatch_index + 1, n_train_batches,
						test_score * 100.))

			if patience <= iter:
				done_looping = True
				break

	end_time = time.clock()
	print(('Optimization complete.  Best validation score of %f %% '
                   'obtained at iteration %i, with test performance %f %%') %
                   (best_validation_loss * 100., best_iter + 1, test_score * 100.))
	print >> sys.stderr, ('The code for file ' +
                                                        os.path.split(__file__)[1] +
                                                        ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
	drive_mlp()
