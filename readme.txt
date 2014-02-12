SdA_reduce is an algorithm for performing dimensionality reduction.  The model used to do the redution is a Stacked de-noising Autoencoder (see: http://bit.ly/1fYYjZO).  It is not as simple to use or as quick to train and apply as existing alternatives such as PCA, but there are advantages to be gained from applying a bit more work and computing power (see: http://bit.ly/1aUVfAr)

The algorithm has four main steps:

(1) Unsupervised model search: given the number of layers and range of layer sizes, perform a model search, training the parameters of each layer sequentially.

(2) Hyper-parameter tuning: perform a grid search over hyper-parameter values using the pre-trainined models.

(3) Model fine-tuning: fine-tune each model by minimizing the reconstruction error, using the hyper-parameter values and pre-trained models.

(4) Reduce: compute the low-dimensional representation of new data using the fine-tuned models.








		
 
