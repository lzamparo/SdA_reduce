""" Utilities for data set extraction and scaling """
from tables import *
import numpy as np



""" Take a reference to an open hdf5 pytables file, extract the first num_files chunks, stack 
them together and return the larger nparray.  Also extract the labels, return them. """
def extract_labeled_chunkrange(data_set_file, num_files = 1):
    arrays_list = data_set_file.listNodes("/recarrays", classname='Array')
    labels_list = data_set_file.listNodes("/labels", classname='Array')
    data = np.empty(arrays_list[0].shape)
    labels = np.empty(labels_list[0].shape)
    
    if num_files > len(arrays_list):
        print "Error!  Asking for %d data files when only %d are available" % (num_files, len(arrays_list))
        return None
    
    empty = True
    for (datanode, labelnode) in zip(arrays_list[0:num_files],labels_list[0:num_files]):
        if empty:
            data[:] = datanode.read()
            labels[:] = labelnode.read()
            empty = False
        else:
            data = np.vstack((data,datanode.read()))
            labels = np.vstack((labels,labelnode.read()))
            
    return data, labels

""" Take a reference to an open hdf5 pytables file, extract the first num_files chunks, stack 
them together and return the larger nparray."""
def extract_unlabeled_chunkrange(data_set_file, num_files = 1):
    
    arrays_list = data_set_file.listNodes("/recarrays", classname='Array')
    
    if num_files > len(arrays_list):
        print "Error!  Asking for more data than is available"
        return None
    
    empty = True
    data = np.empty(arrays_list[0].shape)
    
    for datanode in arrays_list[0:num_files]:
        if empty:
            data[:] = datanode.read()
            empty = False
        else:
            data = np.vstack((data,datanode.read()))
            
    return data

""" Take a reference to an open hdf5 pytables file, extract the specified chunk which corresponds to an element in arrays_list, return as an nparray. """
def extract_unlabeled_byarray(data_set_file, chunk = 1):
    arrays_list = data_set_file.listNodes("/recarrays", classname='Array')
    
    if chunk > len(arrays_list):
        print "Error!  Asking for more data than is available"
        return None
    
    data = np.empty(arrays_list[chunk].shape)
    data[:] = arrays_list[chunk].read()
               
    return data


""" Take a reference to an open hdf5 pytables file, extract the specified chunk of data and corresponding labels, return as nparrays. """
def extract_labeled_byarray(data_set_file, chunk = 1):
    arrays_list = data_set_file.listNodes("/recarrays", classname='Array')
    labels_list = data_set_file.listNodes("/labels", classname='Array')
    
    if chunk > len(arrays_list):
        print "Error!  Asking for more data than is available"
        return None
    
    data = np.empty(arrays_list[chunk].shape)
    labels = np.empty(labels_list[chunk].shape)
    data[:] = datanode.read()
    labels[:] = labelnode.read()
                
    return data, labels

