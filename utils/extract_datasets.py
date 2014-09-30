""" Utilities for data set extraction and scaling """
from tables import *
import numpy as np


def extract_labeled_chunkrange(data_set_file, num_files = 1, offset = 0):
    """ Take a reference to an open hdf5 pytables file, extract the first num_files chunks, stack 
    them together and return the larger nparray.  Also extract the labels, return them. """    
    arrays_list = data_set_file.listNodes("/recarrays", classname='Array')
    labels_list = data_set_file.listNodes("/labels", classname='Array')
    
    
    if num_files > len(arrays_list):
        print("Error!  Asking for", num_files, "data files when only ", len(arrays_list), "are available") 
        return None
    
    if num_files + offset > len(arrays_list):
        print("Error!  Asking for", num_files, "data files beginning at", offset, "but only ", (len(arrays_list) - offset),"are available from start to end")
        return None
        
    empty = True
    start = offset 
    end = offset + num_files
    data = np.empty(arrays_list[start].shape)
    labels = np.empty(labels_list[start].shape)    
    
    for (datanode, labelnode) in zip(arrays_list[start:end],labels_list[start:end]):
        if empty:
            data[:] = datanode.read()
            labels[:] = labelnode.read()
            empty = False
        else:
            data = np.vstack((data,datanode.read()))
            labels = np.vstack((labels,labelnode.read()))
            
    return data, labels


def extract_unlabeled_chunkrange(data_set_file, num_files = 1, offset = 0):
    """ Take a reference to an open hdf5 pytables file, extract the first num_files chunks, stack 
    them together and return the larger nparray."""    
    
    arrays_list = data_set_file.listNodes("/recarrays", classname='Array')
    
    if num_files > len(arrays_list):
        print("Error!  Asking for more data than is available")
        return None
    
    if num_files + offset > len(arrays_list):
        print("Error!  Asking for", num_files, "data files beginning at", offset, "but only ", (len(arrays_list) - offset),"are available from start to end")
        return None    
    
    empty = True
    start = offset 
    end = offset + num_files    
    data = np.empty(arrays_list[start].shape)
    
    for datanode in arrays_list[start:end]:
        if empty:
            data[:] = datanode.read()
            empty = False
        else:
            data = np.vstack((data,datanode.read()))
            
    return data


def extract_unlabeled_byarray(data_set_file, chunk = 1):
    """ Take a reference to an open hdf5 pytables file, extract the specified chunk which corresponds to an element in arrays_list, return as an nparray. """
    arrays_list = data_set_file.listNodes("/recarrays", classname='Array')
    
    if chunk > len(arrays_list):
        print("Error!  Asking for more data than is available")
        return None
    
    data = np.empty(arrays_list[chunk].shape)
    data[:] = arrays_list[chunk].read()
               
    return data



def extract_labeled_byarray(data_set_file, chunk = 1):
    """ Take a reference to an open hdf5 pytables file, extract the specified chunk of data and corresponding labels, return as nparrays. """
    arrays_list = data_set_file.listNodes("/recarrays", classname='Array')
    labels_list = data_set_file.listNodes("/labels", classname='Array')
    
    if chunk > len(arrays_list):
        print("Error!  Asking for more data than is available")
        return None
    
    data = np.empty(arrays_list[chunk].shape)
    labels = np.empty(labels_list[chunk].shape)
    data[:] = datanode.read()
    labels[:] = labelnode.read()
                
    return data, labels


def store_unlabeled_byarray(data_set_file, arrays_group, zlib_filters, data_range, my_data):
    """ Take a reference to an open hdf5 pytables file, and a numpy array, store the numpy array in the specified file. 
            
        :type data_set_file: pytables file reference
        :param data_set_file: an open hdf5 file
    
        :type arrays_group: string
        :param arrays_group: the group of where to write the data chunk
        
        :type zlib_filters: Filter
        :param zlib_filters: The pytables filter to apply
        
        :type data_range: string
        :param data_range: The name of this data chunk
        
        :type my_data: numpy array
        :param my_data: The numpy array to write to the hdf5 file
    """    
    atom = Atom.from_dtype(my_data.dtype)
    ds = data_set_file.createCArray(where=arrays_group, name=data_range, atom=atom, shape=my_data.shape, filters=zlib_filters)
    ds[:] = my_data
    data_set_file.flush()