#-*- coding: UTF-8 -*-
import numpy as np
import caffe
import matplotlib.pyplot as plt
from get_test_data import *

x,y = get_data()    
caffe.set_device(0)
caffe.set_mode_gpu()
mu = np.array([125.306918047,122.950394141,113.865383184])
#mu = np.load('/data/hjy1312/data/RESNET/cifar-10/mean.npy')
#mu = mu.mean(1).mean(1)
model_def = 'deploy.prototxt'
model_prefix = './snapshot/_iter_'
model_postfix = '.caffemodel'
model_index = range(2000,102000,2000)
best_model_index = 0
best_acc = 0.0
acc_set = []
#labels_filename = 'labels.txt'
#labels = np.loadtxt(labels_filename, str, delimiter='\t')
for index in model_index:
    caffe_model = model_prefix+str(int(index))+model_postfix
    net = caffe.Net(model_def, caffe_model, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', mu) 
    #transformer.set_raw_scale('data', 255)
    #transformer.set_channel_swap('data', (2,1,0))
    right = 0
    for i in xrange(x.shape[0]):
        im = x[i]
        net.blobs['data'].data[...] = transformer.preprocess('data',im)
        out = net.forward()
        prob = net.blobs['InnerProduct1'].data[0].flatten()
        order = prob.argsort()[-1]
        print order,y[i]
        if(order==y[i]):
            right = right+1
    acc = float(right)/float(x.shape[0])#very necessary, otherwise the acc will be 0
    acc_set = acc_set + [acc]
    if(acc>best_acc):
        best_acc = acc
        best_model_index = index
print 'the best acc is',best_acc
print 'the best model index is',best_model_index
acc_set = np.array(acc_set)
model_index = np.array(model_index)
plt.plot(model_index,acc_set)
plt.xlabel('model')
plt.ylabel('accuracy')
plt.title('Tesing Accuracy')
plt.show()


"""
caffe_model = model_prefix+str(int(66000))+model_postfix
net = caffe.Net(model_def, caffe_model, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mu) 
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))
im = caffe.io.load_image(img)
net.blobs['data'].data[...] = transformer.preprocess('data',im)
out = net.forward()
labels = np.loadtxt(labels_filename, str, delimiter='\t')
prob = net.blobs['Softmax1'].data[0].flatten()
order = prob.argsort()[-1]
plt.title(labels[order])
plt.imshow(im)
plt.axis('off')
plt.show()
print 'the class is',labels[order]
for layer_name,blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)
print mu
"""
