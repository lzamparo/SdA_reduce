#! /usr/env/python

import os, sys
import re
from numpy import log
import pandas as pd

# cd into the tld
tld = '/data/dA_results'
currdir = os.listdir('.')

# ignore the current directory ('.')
ignore = ['.']

# the RE for extracting corruption params from filenames
corruption_re = re.compile('^([\w]{2,4})\_[\w|\.]+\_([\d\.]{3,4})[\_|\.]')

# extract the final reconstruction error score and corruption values for the given file
def parse_file(filename, dirname):
    with open(os.path.join(dirname,filename), mode='r', buffering=1) as f:
        unit,corruption = strip_corruption(filename, corruption_re)
        lines = f.readlines()
        score = []
        for line in lines[1:-1]:
            line = line.strip()
            parts = line.split()
            score.append(float(parts[-1]))     
    return unit, score, corruption


# parse the filenames for recovering corruption parameters
def strip_corruption(filename,regex):
    match = regex.match(filename)
    if match is not None:
        if len(match.groups()) < 3:
            return match.groups()[0],match.groups()[1]
        else:
            return None 

# set up initial results dict
results = {'gb': {}, 'relu': {}}

# for dirname, dirlist, filenames in os.walk('.'):
print "....parsing the results: "
for dirpath, dirnames, filenames in os.walk(tld):
    if dirpath == '.':
        continue
    for infile in filenames:
        unit, score, corruption = parse_file(infile,dirpath)
        results[unit][corruption] = score

# transform dicts into DFs
print "....transforming nested dict into DFs: "

df_list = []
for unit in results.keys():
    for corruption_val in results[unit].keys():
        this_df = pd.DataFrame(data={'Activation': [unit for i in results[unit][corruption_val]], 
                                     'Corruption': [corruption_val for i in results[unit][corruption_val]],
                                     'Value': results[unit][corruption_val],
                                     'Epoch': range(0,50)}, index=None, columns=None, dtype=None, copy=False)
        df_list.append(this_df)

# concatenate dfs together, write out to file
result = pd.concat(df_list)
result.to_csv(os.path.join(tld,"noise_df.csv"),index=False)
