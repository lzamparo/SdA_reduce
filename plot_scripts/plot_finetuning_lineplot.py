""" Process all the model finetuning output files in the given directory
and produce a lineplot of the top 10 models based on reconstruction error.

Files names look like this: finetune_sda_900_500_100_50.2013-06-16.02:25:26.981658

The average training error for each layer of each model is reported:

e.g epoch 198, minibatch 59/59, validation error 424.194899 %

Each layer transition is marked by the line: Pickling the model..."""

import sys, re, os
import numpy as np
from collections import OrderedDict

from pylab import savefig, figure, plot, xticks, xlabel, ylabel, title, legend

# Extract the model name from each filename.
def extract_model_name(regex,filename):
    match = regex.match(filename)
    if match is not None:
        return match.groups()[0]

# Extract the layer and cost from a line
def extract_cost(line):
    parts = line.split(" ")
    cost = 9999.
    try:
        cost = float(parts[-2])
    except:
        print "Value error in casting " + str(parts[-2]) + " to float"
    return cost

# get the directory from sys.argv[1]
input_dir = sys.argv[1]

# read a list of all files in the directory that match model output files
currdir = os.getcwd()
os.chdir(input_dir)
model_files = os.listdir(".")

# compile a regex to extract the model from a given filename
model_name = re.compile(".*?sda_([\d_]+)\.*")

# Store the results of the model search in this dictionary
# keys are model name, values are validation scores
results = OrderedDict()

print "...Processing files"

# for each file: 
for f in model_files:
    # if this file is a pkl or other file, ignore it
    if not f.startswith("finetune_sda"):
        continue
    
    # read the file, populate this file's entry in the three dicts
    f_model = extract_model_name(model_name, f)
    if f_model is None:
        continue
    
    if not results.has_key(f_model):
        results[f_model]= []
        
    infile = open(f, 'r')
    for line in infile:
        if not line.startswith("epoch"):
            continue
        cost = extract_cost(line)
        results[f_model] = results[f_model] + [cost]
    infile.close()
    
print "...Done"
    
print "...Sorting models by best validation error"
minvals = {key: min(results[key]) for key in results.keys()}
sorted_results = sorted(minvals.items(), key=lambda t: t[1])
top_models = [model for (model, score) in sorted_results]

figure(1)
print "....Top five archs: "
colours = ['r','g','b','k','m']
styles = ['--',':',':',':',':']
plot_objs = [None,None,None,None,None]

for i in range(0,5):
    
    # plotted results
    plot_objs[i], = plot(results[top_models[i]],linewidth=0.8,linestyle='--',color=colours[i])
     
    # printed results
    model, score = sorted_results[i]
    print str(i) + ": " + model + " , " + str(score)    

    
xlabel('Training Epoch')
ylabel('Reconstruction Error')
title('Fine tuning of top-5 performing SdA models')    
legend( tuple(plot_objs), tuple([model.replace('_','-') for model in top_models]), 'upper right', shadow=True, fancybox=True)

os.chdir(currdir)
savefig("finetuning_top5.pdf", dpi=100, format="pdf")

print "....Bottom five archs: "

maxvals = {key: max(results[key]) for key in results.keys()}
sorted_results = sorted(minvals.items(), key=lambda t: t[1],reverse=True)
top_models = [model for (model, score) in sorted_results]

figure(2)
for i in range(1,6):
    
    # plotted results
    plot_objs[i-1], = plot(results[top_models[i]],linewidth=0.8,linestyle='--',color=colours[i-1])
     
    # printed results
    model, score = sorted_results[i]
    print str(i) + ": " + model + " , " + str(score)    

    
xlabel('Training Epoch')
ylabel('Reconstruction Error')
title('Fine tuning of bottom-5 performing SdA models')    
legend( tuple(plot_objs), tuple([model.replace('_','-') for model in top_models]), 'upper right', shadow=True, fancybox=True)

sl_max = max(minvals.values())
sl_min = min(minvals.values())
print "Max, min, mean " + ": " + str(sl_max) + " , " + str(sl_min) + " , " + str(np.mean(minvals.values()))

savefig("finetuning_bottom5.pdf", dpi=100, format="pdf")

