"""
 This script fine-tunes a stacked de-noising auto-encoder by minimizing reconstruction error 
 over larger mini-batches of data.
  

 References :

 http://deeplearning.net/tutorials/ 
 
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""
import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from extract_datasets import extract_unlabeled_chunkrange
from load_shared import load_data_unlabeled
from tables import openFile

from SdA import SdA

from datetime import datetime
from optparse import OptionParser
import os



def finetune_SdA(model_file, finetune_lr=0.01, momentum=0.3, weight_decay = 0.0001, finetuning_epochs=10,
             batch_size=100):
    """
    Finetune and validate a pre-trained SdA.

    :type model_file: string
    :param dataset: path the the pickled model file
    
    :type finetune_lr: float
    :param finetune_lr: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type momentum: float
    :param momentum: the weight given to the previous update when 
    calculating the present update to the weights
    
    :type weight_decay: float
    :param weight_decay: the degree to which updates are degraded in each update.  
    Acts as a regularizer against large weights.

    :type finetuning_epochs: int
    :param finetuning_epochs: number of epoch to do finetuning
    
    :type batch_size: int
    :param batch_size: size of the mini-batches 

    """       
    
    # TODO: error checking here using 'with' ?

    # Get the training, validation data samples from the input file
    data_set_file = openFile(str(options.inputfile), mode = 'r')
    datafiles = extract_unlabeled_chunkrange(data_set_file, num_files = 15, offset = options.offset)
    train_set_x = load_data_unlabeled(datafiles)
    validation_datafiles = extract_unlabeled_chunkrange(data_set_file, num_files = 5, offset = options.offset + 16)
    valid_set_x = load_data_unlabeled(validation_datafiles)
    data_set_file.close()

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    print '... loading the model'
        
    # TODO: error checking here using 'with'?    
    f = file(options.model, 'rb')
    sda = cPickle.load(f)
    f.close()   
    
    
    # set up text output file
    current_dir = os.getcwd()    
    
    os.chdir(options.dir)
    today = datetime.today()
    day = str(today.date())
    hour = str(today.time())
    output_filename = "finetuning_sda." + day + "." + hour
    output_file = open(output_filename,'w')
    
    print >> output_file, "Run on " + str(datetime.now())    
    
    os.chdir(current_dir)    
    
    
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation function for the model
    datasets = (train_set_x,valid_set_x)
    print '... getting the finetuning functions'
    train_fn, validate_model = sda.build_finetune_functions_reconstruction(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)

    print '... fine-tuning the model'
    
    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < finetuning_epochs) and (not done_looping):
        epoch = epoch + 1
        
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print >> output_file, ('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print >> output_file, (('Optimization complete with best validation score of %f ') %
                 (best_validation_loss))
    print >> output_file, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-m", "--model", dest="model", help="load the model from this model file.")
    parser.add_option("-d", "--dir", dest="dir", help="write model output (pkl and txt) to this directory.")
    parser.add_option("-i", "--inputfile", dest="inputfile", help="the data (hdf5 file) prepended with an absolute path")
    parser.add_option("-o", "--offset", dest="offset", type='int', help="use this offset when drawing data.")
    parser.add_option("-c", "--corruption", dest="corruption", type='float', help="use this level of corruption for the denoising AE.")
    
        
    (options, args) = parser.parse_args()    
    finetune_SdA(model_file=options.model)
