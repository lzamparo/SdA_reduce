"""
Utilities for metric learning code
"""

import numpy as np
import warnings

def labels_to_constraints(X, labels, dtype=float, s_size=50, d_size=50, s_delta=0.1, d_delta=1.0):
    """ 
    Convert the (row or column) class labels of type dtype into two sets set of constraints:
    The set S of similarity constraints, and the set D of dis-similarity constraints.  
    """
    
    # Examine lables, to determine (a) if it is sorted by class and (b) where those borders lie.
    class_alphabet = np.unique(labels)
    indicators = np.zeros((class_alphabet.shape[0],labels.shape[0]))
    for endpt in class_alphabet:
        indicators[:,endpt-1] = np.nonzero(labels[:,0] == endpt)[0]

    # Construct the similarities set S
    # perform stratified sampling on X by label, then select within-class pairs with distance <= s_delta
    similar_constraints = sample_similar(X, indicators, s_size, s_delta)
    
    # Construct the differences set D
    # perform stratified sampling on X by label, select differing class pairs withi distance > d_delta
    difference_constraints = sample_differences(X, indicators, d_size, d_delta)
    
    return (similar_contraints,difference_constraints)

def sample_similar(X, indicators, set_size, tolerance):
    """
    Sample points at random from the same tranche of classes contained in 'indicators'.  Build the set of similarities in a recarray.
    """
    #wt_data = X[wt_labels,5:]
    pass

def sample_differences(X, indicators, set_size, tolerance):
    """
    Sample points from different tranches of classes contained in 'indicators'.  Build the set of differences in a rec array
    """
    pass
