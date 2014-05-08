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

# Shared variables for simulation
maxval = shared(numpy.asarray(options.max, config.floatX))
maxval_h = maxval.get_value(borrow=True)
x = shared(numpy.asarray(10 * rng.rand(vlen), config.floatX).reshape((1000,1000)))
u = shared(numpy.asarray(10 * rng.rand(vlen), config.floatX).reshape((1000,1000)))

print >> output_file, "Inital 1-Norm of the X matrix is: ", str(norm(x.get_value(borrow=True), ord=1))

# function to simulate a parameter update to a matrix
add_update = function([], x + u, updates={x: x+u})

# expressions & function to simulate sum of all squares calc
squares = x**2
cumulative_sum = squares.sum()
ss = function([], outputs=cumulative_sum)

# function to rescale the matrix 
val = T.scalar(name="scale_value", dtype=config.floatX)
rescale_x = function(inputs=[val],outputs = x, updates={x: x * val})

for i in xrange(iters):
    r = add_update()
    sfactor = ss()
    scale = maxval_h / numpy.amax([maxval_h,sfactor])
    print >> output_file, "Scale factor is: ", str(scale)
    print >> output_file, "1-Norm of the updated matrix X is: ", str(norm(r, ord=1))
    xval = rescale_x(scale)
    print >> output_file, "1-Norm of the re-scaled matrix X is: ", str(norm(x.get_value(borrow=True), ord=1))    
    

output_file.close()
