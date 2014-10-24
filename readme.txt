SdA_reduce implements a pure Stacked De-noising Autoencoder model for performing dimensionality reduction  (for more information on SdA see: http://bit.ly/1fYYjZO).  It is not as simple to use or as quick to train and apply as existing alternatives such as PCA, but there are advantages to be gained from applying a bit more work and computing power (see: http://bit.ly/1aUVfAr)

SdA_reduce offers several different options for constructing different aspects of your model.  There are different activation functions, regularizers, and SGD algorithms.  The code is research grade, so if it breaks don't be surprised.  I've tested it, both in whole and in part, on Python 2.7.3 with Theano 0.6 RC3.

SdA_reduce has four main steps:

1) Unsupervised model search: given the number of layers and range of layer sizes, perform a model search, training the parameters of each layer sequentially.
2) Hyper-parameter tuning: perform a grid search over hyper-parameter values using the pre-trainined models.
3) Model fine-tuning: fine-tune each model by minimizing the reconstruction error, using the hyper-parameter values and pre-trained models.
4) Reduce: compute the low-dimensional representation of new data using the fine-tuned models.

Some of the scripts in the existing codebase are hardcoded to use the data produced by my collaborators.  To use your own data, you'll have to:

1) Compile it into an hdf5 file
2) Write your own functions for extracting the data sets.  Look at utils/extract_datasets.py for how I do this.
3) Write your own job submission script or batch script for running each of the four main steps above.  Email me if you'd like an example.






		
 
