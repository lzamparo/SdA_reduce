""" Utilities for data set extraction """
from tables import *
import numpy as np

""" Take a reference to an open hdf5 pytables file, extract num_files first nodes, stack 
them together and return the larger nparray.  Also extract the labels, return them. """
def extract_datasets(data_set_file, num_files = 3):
    arrays_list = data_set_file.listNodes("/recarrays", classname='Array')
    labels_list = data_set_file.listNodes("/labels", classname='Array')
    data = np.empty(arrays_list[0].shape)
    labels = np.empty(labels_list[0].shape)
    
    if num_files > len(arrays_list):
        print "Error!  Asking for more data than is available"
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


