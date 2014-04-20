from multiprocessing import Process, Manager
from optparse import OptionParser
import os, re

import cPickle
import gzip
import sys
import time

import numpy
from scipy.linalg import norm

from extract_datasets import extract_unlabeled_chunkrange
from load_shared import load_data_unlabeled
from tables import openFile

from datetime import datetime

def test_gradient_SdA(shared_args,private_args,finetune_lr=0.01, momentum=0.3, weight_decay = 0.0001, finetuning_epochs=5,
             batch_size=1000):
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
    output_filename = "test_gradient_SdA_" + private_args['arch'] + "." + day + "." + hour
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
    #DEBUG
    print >> output_file, "elements, features are: " + str(n_train_batches) + ", " + str(n_features)
    n_train_batches /= batch_size
    
    print >> output_file, "number of training batches: " + str(n_train_batches)
    
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
    
    train_fn, validate_model = sda.build_finetune_functions_reconstruction(
                    datasets=datasets, batch_size=batch_size,
                    learning_rate=finetune_lr)    
    
    # validate every epoch for testing
    validation_frequency = 1
    
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
                    print >> output_file, ('epoch %i, minibatch %i/%i, validation error %f ' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss))
    
                    # DEBUG: test the gradient at some batch value
                    # Arbitrarily picking the first 100 points in the validation set.
                    eval_grad = sda.test_gradient(valid_set_x)
                    grad_vals = [eval_grad(i) for i in xrange(100)]        
                    grad_vals_frob = [norm(A) for A in grad_vals]    
                    grad_vald_one = [norm(A, ord='1') for A in grad_vals]
                    print >> output_file, ('Norm of gradient vals: mean Frobenius %f , mean Max %f' %
                                          (numpy.mean(grad_vals_frob),numpy.mean(grad_vals_one)))                    
                    
    end_time = time.clock()
    print >> output_file, (('Optimization complete with best validation score of %f ') %
                     (best_validation_loss))
    print >> output_file, ('The training code for file ' + os.path.split(__file__)[1] +
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
    (options, args) = parser.parse_args()    
    
    # Construct a dict of shared arguments that should be read by both processes
    manager = Manager()

    args = manager.list()
    args.append({})
    shared_args = args[0]
    shared_args['dir'] = os.path.join(options.dir,options.extension)
    shared_args['input'] = options.inputfile
    shared_args['offset'] = options.offset
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
    p_args['restore'] = pkl_load_file
   
    # Determine where to load & save the second model
    pkl_load_file = os.path.join(parts[0],'pretrain_pkl_files',options.experiment,options.qr_file)
    q_args['restore'] = pkl_load_file

    # Run both sub-processes
    p = Process(target=test_gradient_SdA, args=(args,p_args,))
    q = Process(target=test_gradient_SdA, args=(args,q_args,))
    p.start()
    q.start()
    p.join()
    q.join()