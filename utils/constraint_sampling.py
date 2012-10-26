"""
Utilities for metric learning code
"""

import numpy as np
from scipy.spatial.distance import pdist
import warnings
from numpy.testing import assert_equal
from numpy.random import shuffle, randint

def labels_to_constraints(X, labels, s_size=50, d_size=50, s_delta=0.1, d_delta=1.0):
    """ 
    Take the row major data matrix X, and column vector labels into two sets set of constraints:
    The set S of similarity constraints, and the set D of dis-similarity constraints.  
    """

    # Construct the similarities set S
    # perform stratified sampling on X by label, then select within-class pairs with distance <= s_delta
    similar_constraints = sample_similar(X, labels, s_size, s_delta)
    
    # Construct the differences set D
    # perform stratified sampling on X by label, select differing class pairs withi distance > d_delta
    difference_constraints = sample_differences(X, labels, d_size, d_delta)
    
    return similar_contraints,difference_constraints

def extract_one_class(X,labels,y):
    """ Take an array of data X, a column vector array of labels, and one particular label y.  Return an array
    of all instances in X that have label y
    """

    return X[np.nonzero(labels[:] == y)[0],:]


def estimate_class_sizes(labels):
    """ Take an array of labels, return an array of class size proportions """
    class_alphabet = np.unique(labels)
    n_samples = float(max(labels.shape))
    proportions = np.zeros((1,max(class_alphabet.shape)))
    for position, elem in enumerate(class_alphabet):
        size_set = np.nonzero(labels[:] == elem)[0]
        proportions[0,position] = float(max(size_set.shape)) / n_samples
    return proportions    
        
def draw_pairs(data):
    """ Take a vector of labels, split in half and randomly assign pairs of points to be sampled for similarity matches.
    Return the ndarray of pairs of points. """
    pts = np.arange(data.shape[0])
    first_pts = pts[0:np.floor(data.shape[0]/2)]
    second_pts = pts[np.floor(data.shape[0]/2):]
    common_size = min(first_pts.size,second_pts.size)
    first_pts = first_pts[0:common_size]
    second_pts = second_pts[0:common_size]
    first_pts.shape = (first_pts.size,1)
    second_pts.shape = (second_pts.size,1)
    np.random.shuffle(first_pts)
    np.random.shuffle(second_pts)
    sampling_list = np.hstack((first_pts,second_pts))
    return sampling_list

def generate_points(data, labels, size):
    """ Generate one set of constraints: select two classes  """
    label_alphabet = np.unique(labels)
    assert(label_alphabet.size > 1)
    for useless in xrange(size):
        shuffle(label_alphabet)
        first_class = extract_one_class(data,labels,label_alphabet[0])
        second_class = extract_one_class(data,labels,label_alphabet[1])
        first_pt = first_class[randint(0,first_class.shape[0]),:]
        second_pt = second_class[randint(0,second_class.shape[0]),:]
        pair = np.hstack((first_pt,second_pt))
        for val in pair.ravel(): yield val
    
    
def sample_similar(X, labels, set_size, tolerance):
    """
    Sample points at random from each class, and build the set of similarities in a recarray.
    """
    
    # Examine lables, to determine (a) if it is sorted by class and (b) where those borders lie.
    class_alphabet = np.unique(labels)
    assert_equal(min(class_alphabet),1)
    
    # Calculate the percentages of each set of labels
    proportions = estimate_class_sizes(labels)
    num_examples = np.floor(proportions * set_size)
    
    # Store the similarity points as references to data rows
    similar_pts = zeros(set_size,2 * X.shape[1])
    
    for elem in class_alphabet:
        data = extract_one_class(X,labels,elem)
        this_class_pts = np.zeros(1,2 * data.shape[1])
        
        # Construct a permuted list of potential pairs      
        pairs_list = draw_pairs(data)
        for rowid,row in enumerate(pairs_list):
            dist = pdist(data[row,:])
            if dist[0] < tolerance:
                new_row = np.hstack(data[row[0],:],data[row[1],:])
                this_class_pts = np.vstack(this_class_pts,row)
            
        # Copy this_class_pts into a block in similar_pts, omitting the first zero row
        this_class_pts = this_class_pts[1:,:]
        start_row = min(np.arange(np.all(similar_pts == 0.0, axis = 0).size)[np.nonzero(np.all(similar_pts == 0.0, axis = 0))])
        end_row = start_row + this_class_pts.shape[0]
        similar_pts[start_row:end_row,:] = this_class_pts[:]
            
    return similar_pts
            
def sample_differences(X, labels, set_size, tolerance):
    """
    Sample points from different tranches of classes contained in 'indicators'.  Build the set of differences in a rec array
    """
    
    # Examine lables, to determine (a) if it is sorted by class and (b) where those borders lie.
    class_alphabet = np.unique(labels)
    assert_equal(min(class_alphabet),1)
    
    # Calculate the percentages of each set of labels
    proportions = estimate_class_sizes(labels)
    num_examples = np.floor(proportions * set_size)
    
    # Store the similarity points as references to data rows
    dissimilar_pts = zeros(1,2 * X.shape[1])
    
    # Sample pairs of points from different classes, then add them if they are sufficiently different
    for point in generate_odd_points(X,labels,set_size):
        if pdist(point)[0] > tolerance:
            dissimilar_pts = np.append(dissimilar_pts,point.reshape(1,2 * X.shape[1]),axis=0)
        
    return dissimilar_pts[1:,:]
    

# Temporary debug testing
if __name__ == "__main__":
    
    
    data = np.eye(9)
    labels = np.array([0,0,0,1,1,1,2,2,2])    
    test = np.fromiter(generate_points(data,labels,5),dtype=float).reshape(5,-1)
    print test