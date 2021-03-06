import os  
import cPickle as pickle  
  
import numpy as np      
import sklearn
from sklearn.model_selection import train_test_split  
import sklearn.linear_model

import lmdb  
import caffe   
  
def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(X.shape[0], 3, 32, 32).transpose(0,2,3,1).astype("float")
    #X=X.reshape(10000,-1)
    Y = np.array(Y)
    return X, Y
      
def get_data(file_path = '/data/hjy1312/data/RESNET/cifar-10/cifar-10-batches-py'):
    Xt, yt_f = load_CIFAR_batch(os.path.join(file_path, 'test_batch'))
    return Xt,yt_f
if __name__=='__main__': 
   x,y = get_data()
   print x.shape
   for i in xrange(y.shape[0]):
       print y[i] 
