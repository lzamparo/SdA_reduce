""" Process all the SdA homogeneity tests in the given directory
and produce a ranking of top 10 models based on mean homogeneity results.

Files names look like this: 900_500_100_50.npy

Each .npy file contains an nd-array with shape = (1,#iters) """

import sys, re, os
import numpy as np
from collections import OrderedDict

def parse_dir(input_dir):
    """ Read all .npy file provided in input_dir, output a dict of the mean homogeneity results (value) for each model (key)"""
    os.chdir(input_dir)
    model_files = os.listdir(".")

    # Store the results of the model search in this dictionary
    # keys are model name, values are mean homogeneity scores
    results = {}
    for f in model_files:
        # if this file is not an .npy file, ignore it
        if not f.endswith(".npy"):
            continue
        
        # read the file, populate results dict with mean homogeneity value
        parts = f.split('.')
        f_model = parts[0]
        if f_model is None:
            continue
        
        homog_results = np.load(f)
        results[f_model] = homog_results.mean()       

    return results

def parse_dir_raw(input_dir):
    """ Read all .npy file provided in input_dir, output a dict with an array of the raw homogeneity results (value) for each model (key)"""
    os.chdir(input_dir)
    model_files = os.listdir(".")

    # Store the results of the model search in this dictionary
    # keys are model name, values are mean homogeneity scores
    results = {}
    for f in model_files:
        # if this file is not an .npy file, ignore it
        if not f.endswith(".npy"):
            continue
        
        # read the file, populate results dict with mean homogeneity value
        parts = f.split('.')
        f_model = parts[0].strip('gmm')
        if f_model is None:
            continue
        
        homog_results = np.load(f)
        results[f_model] = homog_results.mean()       

    return results

def parse_dir_meanstd(input_dir):
    """ Read all .npy file provided in input_dir, output two dicts (mean, std) for homogeneity results (values) for each model (key)"""
    os.chdir(input_dir)
    model_files = os.listdir(".")

    # Store the results of the model search in this dictionary
    # keys are model name, values are mean homogeneity scores
    mean_results = OrderedDict()
    std_results = OrderedDict()
    for f in model_files:
        # if this file is not an .npy file, ignore it
        if not f.endswith("gmm.npy"):
            continue
        
        # read the file, populate results dict with mean homogeneity value
        parts = f.split('.')
        f_model = parts[0]
        if f_model is None:
            continue
        
        homog_results = np.load(f)
        mean_results[f_model] = homog_results.mean()
        std_results[f_model] = homog_results.std()

    return mean_results, std_results

def return_top_print(results, n):
    """ Find the top n scoring models in each of the dicts.  Also, compute some order statistics to qualify this list: max, min """
    
    sorted_layer_results = sorted(results.items(), key=lambda t: t[1], reverse=True)
    
    for i in range(0,n):
        model, score = sorted_layer_results[i]
        print str(i) + ": " + model + " , " + str(score)
        
    sl_max = max(results.values())
    sl_min = min(results.values())
    print "Max, min, mean " + ": " + str(sl_max) + " , " + str(sl_min) + " , " + str(np.mean(results.values()))        
    
def return_top(results, n):
    """ Find the top n scoring models in the results dict, return that subset of the results. """
    sorted_layer_results = sorted(results.items(), key=lambda t: t[1], reverse=True)
    return sorted_layer_results[0:n]