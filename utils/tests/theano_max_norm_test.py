from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy

from numpy.linalg import norm
from datetime import datetime

from optparse import OptionParser
import os

parser = OptionParser()
parser.add_option("-d", "--dir", dest="dir", help="test output directory")
parser.add_option("-m", "--max", dest="max", help="max norm value")

(options, args) = parser.parse_args()
vlen = 1000 * 1000 
iters = 50

os.chdir(options.dir)
today = datetime.today()
day = str(today.date())
hour = str(today.time())
output_filename = "max_norm_test." + day + "." + hour
output_file = open(output_filename,'w')

print >> output_file, "Run on " + str(datetime.now())

rng = numpy.random.RandomState(22)
maxval = shared(numpy.asarray(options.max, config.floatX))
x = shared(numpy.asarray(10 * rng.rand(vlen), config.floatX).reshape((1000,1000)))
u = shared(numpy.asarray(10 * rng.rand(vlen), config.floatX).reshape((1000,1000)))

cumulative_sum = T.lscalar('cumulative_sum')
cumulative_sum = T.sum(T.sum(T.sqr(x), axis = 0, keepdims=True)) 
scale = maxval / T.max(maxval,cumulative_sum)
scaleval = shared(0, name="scale value")

f = function([], x + u)
max_norm = function(inputs=[],outputs=x, 
                    updates={x: scale * x, scaleval: scale})

print >> output_file, "Scale factor is: ", str(scaleval.get_value(borrow=True))

for i in xrange(iters):
    r = f()
    mn = max_norm()
    # output tests for x,u, norm here
    print >> output_file, "Scale factor is: ", str(scaleval.get_value(borrow=True))
    print >> output_file, "1-Norm of the matrix columns is: ", str(norm(r, ord=1))

output_file.close()
