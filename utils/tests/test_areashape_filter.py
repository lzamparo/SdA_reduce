from tables import openFile
import sys

import contextlib,time
@contextlib.contextmanager
def timeit():
  t=time.time()
  yield
  print(time.time()-t,"sec")

from extract_datasets import extract_unlabeled_chunkrange, extract_chunk_sizes
from load_shared import test_load_data_unlabeled

# Test the Cells AreaShape feature filtering
sm_rep1_h5 = '/scratch/z/zhaolei/lzamparo/sm_rep1_data/sm_rep1_screen.h5'

# Get the training data sample from the input file
data_set_file = openFile(sm_rep1_h5, mode = 'r')

# Get the chunk sizes
chunks = extract_chunk_sizes(data_set_file)
outfile = open('/scratch/z/zhaolei/lzamparo/figures/as_filter_test.txt', mode='w')
for i,size in enumerate(chunks.cumsum()):
    datafiles = extract_unlabeled_chunkrange(data_set_file, num_files = i+1, offset = 0)
    if datafiles is not None:
        print >> outfile, "expected: ", size, " and got a chunk of size: ", datafiles.shape[0]
        with timeit():
          train_set_x = test_load_data_unlabeled(datafiles)
        print >> outfile, "size of filtered data set is: ", train_set_x.shape[0]
    else:
        print >> outfile, "datafiles is none?  what the heck? "

data_set_file.close()
outfile.close()