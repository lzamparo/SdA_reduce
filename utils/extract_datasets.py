""" Utilities for data set extraction and scaling """
from tables import *
import numpy as np

def draw_reference_population(data_set_file,proportion=0.06,root='/plates',ignore_fewer_than=50):
    """ Walk the tree of plates/<plate>/<well>, drawing a proportionate sample from each well """ 
    
    # do I have to declare data = np.empty()?; data[:] = datanode.read()?
    empty = True
    for node in data_set_file.walk_nodes(root, classname='Array'):
        try:
            data = node.read()
            if data.shape[0] < ignore_fewer_than:
                continue
            up_to = int(np.ceil(data.shape[0] *  proportion))
            data_sample = data[np.random.permutation(data.shape[0])[:up_to],:]
            if empty:
                sample_pop = data_sample
                empty = False
            else:
                sample_pop = np.vstack((sample_pop,data_sample))
        except:
            print "Encountered a problem at this node: " + node._v_name
    return sample_pop        
        

def extract_chunk_sizes(data_set_file):
    """ Return a nd array of data chunk sizes """
    arrays_list = data_set_file.listNodes("/recarrays", classname='Array')
    chunk_sizes = np.zeros((len(arrays_list),),dtype=int)
    for i,dataNode in enumerate(arrays_list):
        chunk_sizes[i] = dataNode.nrows
    return chunk_sizes    

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
    
    if num_files == 0:
        print("Error!  Do you really want 0 files?")
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
    
def store_labeled_byarray(data_set_file, arrays_group, labels_group, zlib_filters, data_range, my_data, my_labels):
    """ Take a reference to an open hdf5 pytables file, and a numpy array, store the numpy array in the specified file. 
            
        :type data_set_file: pytables file reference
        :param data_set_file: an open hdf5 file
    
        :type arrays_group: string
        :param arrays_group: the group under which to write the data chunk
        
        :type labels_group: string
        :param labels_group: the group under which to write the labels chunk
        
        :type zlib_filters: Filter
        :param zlib_filters: The pytables filter to apply
        
        :type data_range: string
        :param data_range: The name of this data chunk, and labels chunk.
        
        :type my_data: numpy array
        :param my_data: The numpy array to write to the hdf5 file
        
        :type my_labels: CArray
        :param my_labels: The CArray directly from the labels hdf5 file
    """ 
    data_atom = Atom.from_dtype(my_data.dtype)
    labels_np = np.empty(my_labels.shape)
    labels_np[:] = my_labels.read()
    labels_atom = Atom.from_dtype(labels_np.dtype)
    ds = data_set_file.createCArray(where=arrays_group, name=data_range, atom=data_atom, shape=my_data.shape, filters=zlib_filters)
    ls = data_set_file.createCArray(where=labels_group, name=data_range, atom=labels_atom, shape=my_labels.shape, filters=zlib_filters)
    ds[:] = my_data
    ls[:] = labels_np
    data_set_file.flush()    