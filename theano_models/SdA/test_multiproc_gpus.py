""" Test script for using two GPUs, one per sub-process,
using the python multiprocessing module.  """

from multiprocessing import Process, Manager
from optparse import OptionParser
import os

def f(shared_args,private_args): 
    import theano.sandbox.cuda
    theano.sandbox.cuda.use(private_args['gpu'])
    
    # Print the shared args, stored as first element in the shared_args list:
    d = shared_args[0]
    for key,value in d.iteritems():
        print "Shared arg " + key + ": " + str(value)
    
    name = os.getpid()
    for key,value in private_args.iteritems():
        print str(name) + " private arg " + key + ": " + value
    
    

if __name__ == '__main__':
    
    parser = OptionParser()
    parser.add_option("-d", "--dir", dest="dir", help="test output directory")
    parser.add_option("-s","--savefile",dest = "savefile", help = "Save the model to this pickle file", default=None)
    parser.add_option("-r","--restorefile",dest = "restorefile", help = "Restore the model from this pickle file", default=None)
    parser.add_option("-i", "--inputfile", dest="inputfile", help="the data (hdf5 file) prepended with an absolute path")
    parser.add_option("-c", "--corruption", dest="corruption", type="float", help="use this amount of corruption for the dA")
    parser.add_option("-o", "--offset", dest="offset", type="int", help="use this offset for reading input from the hdf5 file")
    parser.add_option("-a", "--arch", dest="arch", default = "", help="use this dash separated list to specify the architecture of the SdA.  E.g -a 850-400-50")
    (options, args) = parser.parse_args()    
    
    manager = Manager()

    # Construct a dict of shared arguments that should be read by both processes
    args = manager.list()
    args.append({})
    shared_args = args[0]
    shared_args['dir'] = options.dir
    shared_args['input'] = options.inputfile
    shared_args['corruption'] = options.corruption
    shared_args['offset'] = options.offset
    args[0] = shared_args
    
    # Construct the specific args for each process
    p_args = {}
    q_args = {}
    p_args['arch'] = options.arch
    p_args['save'] = options.savefile
    p_args['gpu'] = 'gpu0'
    q_args['arch'] = options.arch
    q_args['save'] = options.savefile
    q_args['gpu'] = 'gpu1'

    # Run both sub-processes
    p = Process(target=f, args=(args,p_args,))
    q = Process(target=f, args=(args,q_args,))
    p.start()
    q.start()
    p.join()
    q.join()

    