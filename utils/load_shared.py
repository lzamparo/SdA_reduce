""" Utilities for normalizing data sets drawn from hdf5 files, scaling those data sets, and returning in 
theano shared variables.  The default values for features to be trimmed correspond to the useful object features
from the foci data set pipeline. """

import numpy as np
from sklearn.preprocessing import scale

import pickle as pkl

@contextmanager
def opened_w_error(filename, mode="r"):
    try:
        f = open(filename, mode)
    except IOError, err:
        yield None, err
    else:
        try:
            yield f, None
        finally:
            f.close()

def apply_constraints(data,constraints_file):
    """ Read constraints from constraints file, filter rows that do not satisfy the constraints, return the array """
    with opened_w_error(constraints_file, mode='rb') as (filename,err):
        if err:
            print('IOError ', err)
            raise err
        else:
            zipped_headers, thresholds = pkl.load(filename)
            
    #Pare away rows that do not satisfy the constraints
    for i,position_tup in enumerate(zipped_headers):
        position, name = position_tup
        lower, upper = thresholds[name]
        data = data[(data[:,position] > lower) & (data[:,position] < upper)]
    return data


def load_data_unlabeled(dataset, features = (5,916), borrow=True, do_filter=False, constraints='/scratch/z/zhaolei/lzamparo/sm_rep1_data/Cells_thresholds.pkl'):
    """ Take an unpacked dataset (from extract_datasets), scale it, and return as a shared theano variable.
    
    :type dataset: numpy ndarray
    :param dataset: the numpy ndarray returned from some function in extract_dataset
    
    :type features: tuple
    :param features: keep only those features indexed between features[0],features[1]  """
    import theano
    
    if do_filter:
        data_filtered = apply_constraints(dataset, constraints)
        
    else:
        data_filtered = dataset
    
    # Scale the data: centre, and unit-var.
    data_scaled = scale(data_filtered)
    
    # if features tuple is defined, throw away unwanted columns
    if features:
        data_scaled = data_scaled[:,features[0]:features[1]]
        
    print '... loading data'
    print '... converting to shared vars'
    
    return theano.shared(np.asarray(data_scaled, dtype=theano.config.floatX), borrow=borrow)    
    

def test_load_data_unlabeled(dataset, features = (5,916), do_filter=True, constraints='/scratch/z/zhaolei/lzamparo/sm_rep1_data/Cells_thresholds.pkl'):
    print '... applying Area Shape filters to the dataset'
    data_filtered = apply_constraints(dataset, constraints)
    
    # Scale the data: centre, and unit-var.
    print '... scaling the data set'
    data_scaled = scale(data_filtered)
    
    # if features tuple is defined, throw away unwanted columns
    if features:
        data_scaled = data_scaled[:,features[0]:features[1]]
        
    return data_scaled

def load_data_labeled(dataset, labels, ratios = np.array([0.8,0.1,0.1]), features = (5,916), do_filter=False, constraints='/scratch/z/zhaolei/lzamparo/sm_rep1_data/Cells_thresholds.pkl'):
    ''' Take an unpacked dataset (from extract_datasets), scale it, and return 
    as shared theano variables.  The form of the returned data is meant to mimic the form MNIST data
    is packaged and returned in load_data from logistic_sgd, part of the theano tutorial

    :type dataset: numpy ndarray
    :param dataset: the numpy ndarray returned from some function in extract_dataset
    
    :type labels: numpy ndarray
    :param labels: a column vector (n x 1) ndarray of labels.
    
    :type ratios: numpy vector
    :param ratios: the specified ratios of how to split dataset into training, test, and validation data.
    
    :type features: tuple
    :param features: keep only those features indexed between features[0],features[1]
    
    '''

    # Take only the first column of the labels.  The other two are image, object numbers.
    labels_vec = labels[:,0]
    
    if do_filter:
        dataset_augmented = np.hstack((dataset,labels_vec[:,np.newaxis]))
        dataset_filtered = apply_constraints(dataset_augmented,constraints)
        labels_vec = dataset_filtered[:,-1]
        data_scaled = scale(dataset_filtered[:,:-1])
        
    else:
        dataset_filtered = dataset
        data_scaled = scale(dataset)
    
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an numpy.ndarray of 2 dimensions (a matrix)
    # which rows correspond to an example. labels is a
    # numpy.ndarray of 1 dimensions (a vector) that has the same length as
    # the number of rows in the input.   
    
    # Scale the data: centre, and unit-var.
    
    
    # if features tuple is defined, throw away unwanted columns
    if features:
        data_scaled = data_scaled[:,features[0]:features[1]]
    
    print '... loading data'
    
    # Calculate the indices where each split into training, test, validation set will take place
    endpts = data_scaled.shape[0] * ratios
    endpts = endpts.astype(int)
    train_start, train_end = 0, endpts[0] -1
    test_start, test_end = endpts[0], endpts[0] + endpts[1] -1
    valid_start, valid_end = endpts[0] + endpts[1], dataset.shape[0] -1
    
    train_set = (data_scaled[train_start:train_end,:], labels_vec[train_start:train_end])
    test_set = (data_scaled[test_start:test_end,:], labels_vec[test_start:test_end])
    valid_set = (data_scaled[valid_start:valid_end,:], labels_vec[valid_start:valid_end])

    print '... converting to shared vars'

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    
    print '... done'
    return rval

def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        import theano
        from theano.tensor import cast
        
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, cast(shared_y, 'int32')
