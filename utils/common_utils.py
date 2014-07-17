""" Utility functions for parsing arguments from pretraining, finetuning scripts """
import re

### Metadata dict writer
def write_metadata(output_file, meta_dict):
    ''' print each entry of the meta dict into the file tied to handle output_file. '''
    for key in meta_dict:
        print >> output_file, key, meta_dict[key]

def extract_arch(filename, model_regex):
    ''' Return the model architecture of this filename
    Modle filenames look like SdA_1000_500_100_50.pkl'''
    match = model_regex.match(filename)
    if match is not None:    
        return match.groups()[0]
    
def get_arch_list(private_args):
    ''' Grab the string representation of the model architecture, put each layer element in a list
     
         :type private_args: dict
         :param private_args: argument dictionary for the given models ''' 
    arch_str = private_args['arch']
    arch_str_canonical = arch_str.replace('_','-')
    arch_list_str = arch_str_canonical.split("-")
    arch_list = [int(item) for item in arch_list_str]
    if len(arch_list) > 1:
        return arch_list
    else:
        errormsg = 'architecture is improperly specified : ' + arch_list[0] 
        raise ValueError(errormsg)


def parse_dropout(d_string, model):
    ''' Return a list of floats specified in d_string, describing how dropout should be applied.
    Format for d_string is some part
    Or, if d_string == 'none', make sure all components are kept for finetuning. '''
    if d_string is not 'none':
        parts = d_string.split('-')
        assert len(parts) == len(model.split('_'))
        return [float(f) / 100 for f in parts]
    else:
        model_parts = model.split('_')
        return [1.0 for layer in model_parts]

    
def parse_layer_type(layer_str, num_layers):
    ''' Return a list of strings identifying the types of layers to use when constructing this model.  
    Acceptable configurations are: 
        'gaussian': for a Gaussian-Bernoulli SdA
        'bernoulli': for a purely Bernoulli SdA
        'relu': for a ReLU SdA
    
    :type layer_str: string
    :param layer_str: the string representation of the layer type
        
    :type num_layers: int
    :param num_layers: number of layers'''
    
    if layer_str.lower() == 'bernoulli':
        layers = ['bernoulli' for i in xrange(num_layers)]
        return layers
    
    elif layer_str.lower() == 'gaussian':
        layers = ['bernoulli' for i in xrange(num_layers)]
        layers[0] = layer_str.lower()
        return layers
    
    elif layer_str.lower() == 'relu':
        layers = ['relu' for i in xrange(num_layers)]
        return layers
    
    else:
        errormsg = 'incompatible layer type specified : ' + layer_str
        raise ValueError(errormsg)