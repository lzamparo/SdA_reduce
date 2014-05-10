import numpy
import cPickle
import os, re, sys, time

from numpy.linalg import norm

from extract_datasets import extract_unlabeled_chunkrange
from load_shared import load_data_unlabeled
from tables import openFile


from datetime import datetime
from optparse import OptionParser
from multiprocessing import Process, Manager

def test_finetune_SdA(shared_args, private_args, num_epochs=10, finetune_lr=0.0001, lr_decay = 0.98, batch_size=20):
    """
    
    Finetune an SdA model for the given number of training epochs.  The model is reconstructed from a previously pickled model.

    :type shared_args: dict
    :param shared_args: dictionary of arguments shared by both threads
    
    :type private_args: dict
    :param private_args: dictionary of arguments exclusively for this thread

    :type num_epochs: int
    :param num_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type batch_size: int
    :param batch_size: train in mini-batches of this size

    """
    
    # Import sandbox.cuda to bind the specified GPU to this subprocess
    # then import the remaining theano and model modules.
    import theano.sandbox.cuda
    theano.sandbox.cuda.use(private_args['gpu'])
    
    import theano
    import theano.tensor as T
    from theano.tensor.shared_randomstreams import RandomStreams
    
    from SdA import SdA    
    
    shared_args_dict = shared_args[0]
    
    current_dir = os.getcwd()    
    os.chdir(shared_args_dict['dir'])
    today = datetime.today()
    day = str(today.date())
    hour = str(today.time())   
    output_filename = "test_finetune_sda_" + private_args['arch'] + "." + day + "." + hour
    output_file = open(output_filename,'w')
    os.chdir(current_dir)    
    print >> output_file, "Run on " + str(datetime.now())    
    
    # Get the training and validation data samples from the input file
    data_set_file = openFile(str(shared_args_dict['input']), mode = 'r')
    datafiles = extract_unlabeled_chunkrange(data_set_file, num_files = 25, offset = shared_args_dict['offset'])
    train_set_x = load_data_unlabeled(datafiles)
    validation_datafiles = extract_unlabeled_chunkrange(data_set_file, num_files = 5, offset = shared_args_dict['offset'] + 25)
    valid_set_x = load_data_unlabeled(validation_datafiles)    
    data_set_file.close()

    # compute number of minibatches for training, validation and testing
    n_train_batches, n_features = train_set_x.get_value(borrow=True).shape
    n_train_batches /= batch_size
    
    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    
    print >> output_file, 'Unpickling the model from %s ...' % (private_args['restore'])        
    f = file(private_args['restore'], 'rb')
    sda = cPickle.load(f)
    f.close()        
    
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation function for the model
    datasets = (train_set_x,valid_set_x)
    
    # Set the initial value of the learning rate
    learning_rate = theano.shared(numpy.asarray(finetune_lr, 
                                                     dtype=theano.config.floatX)) 
        
    # Function to decrease the learning rate
    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
                        updates={learning_rate: learning_rate * lr_decay})     
    
    print '... getting the finetuning functions'
    train_fn, validate_model = sda.build_finetune_functions_reconstruction(
                datasets=datasets, batch_size=batch_size,
                learning_rate=learning_rate)

    print '... fine-tuning the model'    

    # early-stopping parameters
    patience = 300 * n_train_batches  # look as this many batches regardless
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
    start_time = time.clock()

    done_looping = False
    epoch = 0   
    
    # Set up functions for max norm regularization
    max_norm_regularization_fns = sda.max_norm_regularization()

    while (epoch < num_epochs) and (not done_looping):
        epoch = epoch + 1
        
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(index=minibatch_index, momentum=shared_args_dict['momentum'])
            iter = (epoch - 1) * n_train_batches + minibatch_index

            # apply max-norm regularization
            for i in xrange(sda.n_layers):
                scale = max_norm_regularization_fns[i](norm_limit=shared_args_dict['maxnorm'])
                if scale > 1.0:
                    print >> output_file, "Re-scaling took place w scale value ", str(scale)
            
            if (iter + 1) % validation_frequency == 0:
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
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    
                    # simulate saving the best model that achieved this best loss    
                    print >> output_file, '(Simulated) Pickling the model...'          
                                        
             

            if patience <= iter:
                done_looping = True
                break
        
        decay_learning_rate()

    end_time = time.clock()
    print >> output_file, (('Optimization complete with best validation score of %f ') %
                 (best_validation_loss))
    print >> output_file, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))    

    output_file.close()   
      
def extract_arch(filename, model_regex):
    ''' Return the model architecture of this filename
    Modle filenames look like SdA_1000_500_100_50.pkl'''
    match = model_regex.match(filename)
    if match is not None:    
        return match.groups()[0]    
    
if __name__ == '__main__':
    
    # Parse command line args
    parser = OptionParser()
    parser.add_option("-d", "--dir", dest="dir", help="base output directory")
    parser.add_option("-e", "--pretrain_experiment", dest="experiment", help="directory name containing pre-trained pkl files for this experiment (below the base directory)")
    parser.add_option("-x", "--output_extension", dest="extension", help="output directory name below the base, named for this finetuning experiment")
    parser.add_option("-p","--firstrestorefile",dest = "pr_file", help = "Restore the first model from this pickle file", default=None)
    parser.add_option("-q","--secondrestorefile",dest = "qr_file", help = "Restore the second model from this pickle file", default=None)
    parser.add_option("-i", "--inputfile", dest="inputfile", help="the data (hdf5 file) prepended with an absolute path")
    parser.add_option("-o", "--offset", dest="offset", type="int", help="use this offset for reading input from the hdf5 file")
    parser.add_option("-n","--normlimit",dest = "norm_limit", type=float, help = "limit the norm of each vector in each W matrix to norm_limit")
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
    shared_args['maxnorm'] = options.norm_limit
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
    p = Process(target=test_finetune_SdA, args=(args,p_args,))
    q = Process(target=test_finetune_SdA, args=(args,q_args,))
    p.start()
    q.start()
    p.join()
    q.join()        
    
    
    
    