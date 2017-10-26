#!/bin/bash
/data/hjy1312/caffe-master/build/tools/caffe train --solver=./res_net_solver.prototxt --gpu=5 &>./swish_resnet_cifar10.log&
