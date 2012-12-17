"""Testing for constraint sampling methods"""

import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from nose.tools import with_setup

import sys
sys.path.append('/home/lee/projects/screen_clustering/utils/')
from ..constraint_sampling import extract_one_class, estimate_class_sizes, draw_pairs, generate_points, sample_similar, sample_differences

"""Test fixtures"""

# Multiple elements for each class, balanced
test_data_big = np.eye(9) + np.arange(81).reshape(9,9)
test_labels_big = np.array([1,1,1,2,2,2,3,3,3])
test_output_data_big_one =  test_data_big[0:3,:]
test_output_data_big_two =  test_data_big[3:6,:]
test_output_data_big_three =  test_data_big[6:9,:]

# One element for each class, balanced
test_data_small = np.eye(3) + np.arange(9).reshape(3,3)
test_labels_small = np.array([1,2,3])
test_output_labels_small = np.array([0,1,2])
test_output_data_small_one = test_data_small[0,:]
test_output_data_small_one.resize(1,3)
test_output_data_small_two = test_data_small[1,:]
test_output_data_small_two.resize(1,3)
test_output_data_small_three = test_data_small[2,:]
test_output_data_small_three.resize(1,3)

# Multiple elements for each class, unordered and unbalanced
test_data_uu = np.eye(7) + np.arange(49).reshape(7,7)
test_labels_uu = np.array([1,1,2,2,2,1,1])
test_output_data_uu_one = test_data_uu[np.array([0,1,5,6])]
test_output_data_uu_two = test_data_uu[np.array([2,3,4])]
                    
def test_extract_one_class():
    """ construct a label column vector, see if it produces a sensible matrix """
    
    # One element for each class, balanced
    inds = extract_one_class(test_data_small, test_labels_small,1)
    assert_equal(inds, test_output_data_small_one)
    inds = extract_one_class(test_data_small, test_labels_small,2)
    assert_equal(inds, test_output_data_small_two)
    inds = extract_one_class(test_data_small, test_labels_small,3)
    assert_equal(inds, test_output_data_small_three)    
    
    # Multiple elements for each class, balanced
    inds = extract_one_class(test_data_big, test_labels_big,1)
    assert_equal(inds,test_output_data_big_one)
    inds = extract_one_class(test_data_big, test_labels_big,2)
    assert_equal(inds,test_output_data_big_two)    
    inds = extract_one_class(test_data_big, test_labels_big,3)
    assert_equal(inds,test_output_data_big_three)
    
    # Multiple elements for each class, unordered
    inds = extract_one_class(test_data_uu, test_labels_uu,1)
    assert_equal(inds,test_output_data_uu_one)
    inds = extract_one_class(test_data_uu, test_labels_uu,2)
    assert_equal(inds,test_output_data_uu_two)    

def test_estimate_class_sizes():
    """ see if the test fixture labels produce expected results """
    
    # one element for each class, balanced
    one_each = estimate_class_sizes(test_labels_small)
    assert_array_almost_equal(one_each,np.array([[0.333333,0.333333,0.3333333]]))
    
    # multiple elements for each class, balanced
    mult_balanced = estimate_class_sizes(test_labels_big)
    assert_array_almost_equal(mult_balanced,np.array([[0.3333333,0.3333333,0.33333333]]))
    
    # multiple elements for each class, unbalanced
    mult_unbalanced = estimate_class_sizes(test_labels_uu)
    assert_array_almost_equal(mult_unbalanced,np.array([[0.57,0.43]]),decimal=2)

def test_draw_pairs():
    """ Since this is a randomized algorithm, just test to see if 
    an array in the correct shape is returned.  """
    odd_pairs = draw_pairs(test_data_big)
    assert_equal(odd_pairs.shape,(4,2))
    even_pairs = draw_pairs(test_data_big[0:8,:])
    assert_equal(even_pairs.shape,(4,2))
    small_pairs = draw_pairs(test_data_uu)
    assert_equal(small_pairs.shape, (3,2))
      
def test_generate_points():
    """ Since this is a randomized algorithm, just test to see that the shape
    of the returned array of points is correct.  """
    num_pairs = 5
    test_data_big = np.eye(9) + np.arange(81).reshape(9,9)
    test_labels_big = np.array([1,1,1,2,2,2,3,3,3])
    output = np.fromiter(generate_points(test_data_big,test_labels_big,num_pairs),dtype=float).reshape(2*num_pairs,-1)
    assert_equal(output.shape, (10,9))
    
    # Consider including a test of two rows, and asserting that either 
    # [row 1, row 2] is returned, or [row 2, row 1] is returned

    
def test_sample_similar():
    """ """
    pass


def test_sample_different():
    """ """
    pass        




