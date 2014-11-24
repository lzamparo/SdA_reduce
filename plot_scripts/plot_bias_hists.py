# coding: utf-8
import os, sys, re
import pickle
from pickle import UnpicklingError
from collections import OrderedDict

import matplotlib as mpl
mpl.use('pdf')
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def make_bias_hist(pkl_dir,pkl_files,num_layers):
    """ Grab all the visible & hidden biases from the SdA models in pkl_dir, plot histograms. """
    
    import theano.sandbox.cuda
    theano.sandbox.cuda.use('gpu0')
    import theano
    import theano.tensor as T
    from theano.tensor.shared_randomstreams import RandomStreams
    from theano import shared
    
    from SdA import SdA
    from AutoEncoder import AutoEncoder, BernoulliAutoEncoder, GaussianAutoEncoder, ReluAutoEncoder
    
    ### Grab the data from each pickled model ###
    for pkl in pkl_files:
        path = os.path.join(pkl_dir,pkl)
        print "processing " + pkl + " ..."
        try:
            f = file(path,'rb')
            sda_model = pickle.load(f)
            f.close()
            
            for i,layer in enumerate(sda_model.dA_layers):
                W,bhid,bvis = layer.get_params()
                b_vis[i].append(bvis.get_value(borrow=True))
                b_hid[i].append(bhid.get_value(borrow=True))
        except UnpicklingError:
            print "Caught UnpicklingError trying to unpickle from " + path
        print "done "
    
    # bail if I could not unpickle *any* models
    num_items = np.asarray([len(b_vis[k]) for k in b_vis.keys()])
    assert num_items.sum() > 0            
    
    ### Plot the visible, hidden layer histograms ###
    sns.set_palette("deep", desat=.6)
    sns.set_context(rc={"figure.figsize": (12, 8)})
    
    b_fig = plt.figure()
        
    for key in b_hid:
        data = np.concatenate(b_hid[key])
        sp = b_fig.add_subplot(num_layers,2,key+1)
        sp.hist(data,normed=True, color="#6495ED", alpha=.5)
        title = "Hidden layer bias values: layer " + str(key)
        sp.set_title(title)
    
    for key in b_vis:
        data = np.concatenate(b_vis[key])
        offset = key + len(b_hid.keys()) + 1
        sp = b_fig.add_subplot(num_layers,2,offset)
        sp.hist(data,normed=True, color="#F08080", alpha=.5)
        title = "Visible layer bias values: layer " + str(key)
        sp.set_title(title)
    
    # All on one figure
    #for key in b_hid:
        #bhid_data = np.concatenate(b_hid[key])
        #bvis_data = np.concatenate(b_vis[key])
        #max_data = max(bhid_data.max(),bvis_data.max())
        #min_data = min(bhid_data.min(),bvis_data.min())
        #bins = np.linspace(0, max_data, max_data + 1)
        #sp = b_fig.add_subplot(3,1,key+1)
        #hidden, = sp.hist(bhid_data, bins, normed=True, color="#6495ED", alpha=.5, label='Hidden')
        #visible, = sp.hist(bvis_data, bins, normed=True, color="#F08080", alpha=.5, label='Visible')     
        #title = "Bias values layer " + key
        #sp.legend(handles=[hidden,visible])
        #sp.set_title(title)
     
    b_fig.tight_layout()
    filename = "bias_hist_" + str(num_layers) +"_layers.pdf"
    b_fig.savefig(filename)

if __name__ == "__main__":
    # pass the directory with models, number of layers
    try:
        pkl_dir = sys.argv[1]
        num_layers = int(sys.argv[2])
        layers = re.compile(".*/([\d])\_layers/.*")
        match = layers.match(pkl_dir)
        pkl_dir_layers = int(match.groups()[0])
        assert pkl_dir_layers == num_layers
    except:
        print "usage: python bias_hists.py <pkl_dir> <num_layers>"
        sys.exit()
        
    pkl_files = os.listdir(pkl_dir)
    os.chdir("/scratch/z/zhaolei/lzamparo/figures")
    
    b_vis = OrderedDict((k, []) for k in xrange(num_layers))
    b_hid = OrderedDict((k, []) for k in xrange(num_layers))
    
    make_bias_hist(pkl_dir, pkl_files,num_layers)

    
