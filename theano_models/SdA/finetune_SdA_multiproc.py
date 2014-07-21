""" SdA finetuning script that uses two GPUs, one per sub-process,
via the Python multiprocessing module.  """


# These imports will not trigger any theano GPU binding, so are safe to sit here.
from multiprocessing import Process, Manager
from optparse import OptionParser
import os, re

import cPickle
import time

import numpy

from extract_datasets import extract_unlabeled_chunkrange
from load_shared import load_data_unlabeled
from common_utils import extract_arch, parse_dropout, write_metadata
from tables import openFile

from datetime import datetime


def finetune_SdA(shared_args, private_args, finetune_lr=0.001, max_momentum=0.85, finetuning_epochs=300, lr_decay=0.99,
             batch_size=100): 
    """ Finetune and validate a pre-trained SdA for the given number of training epochs.
    Batch size and finetuning epochs default values are picked to roughly match the reported values
    of Hinton & Salakhtudinov.   

    
    :type finetune_lr: float
    :param finetune_lr: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type max_momentum: float
    :param momentum: the maximum value for momentum given to the previous update when 
    calculating the present update to the weights
    
    :type lr_decay: float
    :param lr_decay: the rate at which the learning rate decays after each epoch

    :type finetuning_epochs: int
    :param finetuning_epochs: number of epoch to do finetuning
    
    :type batch_size: int
    :param batch_size: size of the mini-batches
    
    """
    
    # Import sandbox.cuda to bind the specified GPU to this subprocess
    # then import the remaining theano and model modules.
    import theano.sandbox.cuda
    theano.sandbox.cuda.use(private_args['gpu'])
    
    import theano
    import theano.tensor as T
    
    from SdA import SdA    
    
    shared_args_dict = shared_args[0]
    
    current_dir = os.getcwd()    
    os.chdir(shared_args_dict['dir'])
    today = datetime.today()
    day = str(today.date())
    hour = str(today.time())   
    output_filename = "finetune_sda_" + private_args['arch'] + "." + day + "." + hour
    output_file = open(output_filename,'w')
    os.chdir(current_dir)    
    print >> output_file, "Run on " + str(datetime.now())    
    
    # Get the training and validation data samples from the input file
    data_set_file = openFile(str(shared_args_dict['input']), mode = 'r')
    datafiles = extract_unlabeled_chunkrange(data_set_file, num_files = 30, offset = shared_args_dict['offset'])
    train_set_x = load_data_unlabeled(datafiles)
    validation_datafiles = extract_unlabeled_chunkrange(data_set_file, num_files = 5, offset = shared_args_dict['offset'] + 30)
    valid_set_x = load_data_unlabeled(validation_datafiles)    
    data_set_file.close()

    # compute number of minibatches for training, validation and testing
    n_train_batches, n_features = train_set_x.get_value(borrow=True).shape
    n_train_batches /= batch_size
    
    print >> output_file, 'Unpickling the model from %s ...' % (private_args['restore'])        
    f = file(private_args['restore'], 'rb')
    sda = cPickle.load(f)
    f.close()        
    
    print '... writing meta-data to output file'
    metadict = {'n_train_batches': n_train_batches, 'batch_size': batch_size,'finetuning_epochs': finetuning_epochs, 'finetune_lr': finetune_lr}
    metadict = dict(metadict.items() + shared_args_dict.items())
    write_metadata(output_file, metadict)
    
    
    ########################
    # FINETUNING THE MODEL #
    ########################

    # Set the dropout rates, and scale the weights up by the inverse of the dropout rates
    sda.dropout_rates = private_args['dropout']

    # get the training, validation function for the model
    datasets = (train_set_x,valid_set_x)
    
    print '... getting the finetuning functions'
    train_fn, validate_model = sda.build_finetune_full_reconstruction(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr,
                method=shared_args_dict['sgd'])

    print '... fine-tuning the model'    

    # early-stopping parameters
    patience = finetuning_epochs * n_train_batches  # look as this many batches regardless
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; 
                                  # every epoch in this case

    best_params = None
    best_validation_loss = numpy.inf
    done_looping = False
    epoch = 0
    
    # Set the initial value of the learning rate
    learning_rate = theano.shared(numpy.asarray(finetune_lr, 
                                                 dtype=theano.config.floatX)) 
    
    # Function to decrease the learning rate
    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
                    updates={learning_rate: learning_rate * lr_decay})    
    
    # Set up function for max norm regularization
    apply_max_norm_regularization = sda.max_norm_regularization()
    
    # Use NAG ?
    do_NAG = False
    
    # Use weight decay?
    use_wd = shared_args_dict['sgd'].endswith('wd')
    
    start_time = time.clock()

    while (epoch < finetuning_epochs) and (not done_looping):
        epoch = epoch + 1
        
        for minibatch_index in xrange(n_train_batches):
            # Calculate momentum value
            t = (epoch - 1) * n_train_batches + minibatch_index
            momentum = numpy.asarray(min(1 - numpy.power(2,-1 - numpy.log2(numpy.floor(t / finetuning_epochs) +1)), max_momentum),dtype=numpy.float32)          
            
            if do_NAG:
                apply_last_update(momentum)
                
            if use_wd:
                minibatch_avg_cost = train_fn(minibatch_index, momentum, shared_args_dict['weight_decay'])
            else:
                minibatch_avg_cost = train_fn(minibatch_index, momentum)
            
            # DEBUG: monitor the training error
            print >> output_file, ('epoch %i, minibatch %i/%i, training error %f ' %
                    (epoch, minibatch_index + 1, n_train_batches,
                    minibatch_avg_cost))            

            # apply max-norm regularization
            #apply_max_norm_regularization(shared_args_dict['maxnorm'])          

            if (t + 1) % validation_frequency == 0:               
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                
                print >> output_file, ('epoch %i, minibatch %i/%i, validation error %f ' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, t * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = t
                    
                    # save best model that achieved this best loss    
                    print >> output_file, 'Pickling the model...'          
                    f = file(private_args['save'], 'wb')
                    cPickle.dump(sda, f, protocol=cPickle.HIGHEST_PROTOCOL)
                    f.close()                    
                    
            if patience <= t:
                done_looping = True
                break
        # not for adagrad and variants    
        #decay_learning_rate()

    end_time = time.clock()
    print >> output_file, (('Optimization complete with best validation score of %f ') %
                 (best_validation_loss))
    print >> output_file, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))    

    output_file.close()        
    
        
if __name__ == '__main__':
        
    # Parse command line args
    parser = OptionParser()
    parser.add_option("-d", "--dir", dest="dir", help="base output directory")
    parser.add_option("-e", "--pretrain_experiment", dest="experiment", help="directory name containing pre-trained pkl files for this experiment (below the base directory)")
    parser.add_option("-x", "--output_extension", dest="extension", help="output directory name below the base, named for this finetuning experiment")
    parser.add_option("-p","--firstrestorefile", dest="pr_file", help="Restore the first model from this pickle file", default=None)
    parser.add_option("-q","--secondrestorefile", dest="qr_file", help="Restore the second model from this pickle file", default=None)
    parser.add_option("-i", "--inputfile", dest="inputfile", help="the data (hdf5 file) prepended with an absolute path")
    parser.add_option("-o", "--offset", dest="offset", type="int", help="use this offset for reading input from the hdf5 file")
    parser.add_option("-n","--normlimit", dest = "norm_limit", type=float, default=3.0, help="limit the norm of each vector in each W matrix to norm_limit")
    parser.add_option("-u","--dropout", dest="dropout", default="none", help="A dash delimited string describing how dropout should be applied in finetuning, or 'none' for regular finetuning.")
    parser.add_option("-s","--sgdflavour", dest="sgd", default="cm", help="Variant of SGD to employ.  Currently accepting cm, adagrad, adagrad_momentum, cm_wd, adagrad_momentum_wd." )
    
    (options, args) = parser.parse_args()    
    
    # Construct a dict of shared arguments that should be read by both processes
    manager = Manager()

    args = manager.list()
    args.append({})
    shared_args = args[0]
    shared_args['dir'] = os.path.join(options.dir,options.extension)
    shared_args['input'] = options.inputfile
    shared_args['offset'] = options.offset
    shared_args['momentum'] = 0.8
    shared_args['weight_decay'] = 0.0001
    shared_args['maxnorm'] = options.norm_limit
    shared_args['sgd'] = options.sgd
    args[0] = shared_args
    
    # Construct the specific args for each of the two processes
    p_args = {}
    q_args = {}
       
    p_args['gpu'] = 'gpu0'
    q_args['gpu'] = 'gpu1'
    
    # Compile regular expression for extracting model architecture names
    model_name = re.compile(".*?_([\d_]+).pkl")    
    p_args['arch'] = extract_arch(options.pr_file,model_name)
    q_args['arch'] = extract_arch(options.qr_file,model_name)
    
    # Parse the droput parameters for both models
    p_args['dropout'] = parse_dropout(options.dropout, p_args['arch'])
    q_args['dropout'] = parse_dropout(options.dropout, q_args['arch'])
         
    # Determine where to load & save the first model
    parts = os.path.split(options.dir)
    pkl_load_file = os.path.join(parts[0],'pretrain_pkl_files',options.experiment,options.pr_file)
    pkl_save_file = os.path.join(parts[0],'finetune_pkl_files',options.extension,options.pr_file)
    p_args['restore'] = pkl_load_file
    p_args['save'] = pkl_save_file
   
    # Determine where to load & save the second model
    pkl_load_file = os.path.join(parts[0],'pretrain_pkl_files',options.experiment,options.qr_file)
    pkl_save_file = os.path.join(parts[0],'finetune_pkl_files',options.extension,options.qr_file)
    q_args['restore'] = pkl_load_file
    q_args['save'] = pkl_save_file

    # Run both sub-processes
    p = Process(target=finetune_SdA, args=(args,p_args,))
    q = Process(target=finetune_SdA, args=(args,q_args,))
    p.start()
    q.start()
    p.join()
    q.join()

    
