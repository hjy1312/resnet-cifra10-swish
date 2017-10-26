#!/bin/bash
for x in ./snapshot/*.caffemodel
do
	let n++
	/data/hjy1312/caffe-master/build/tools/caffe test --gpu=0  --model=test.prototxt --weights=$x &>./test_log/test$n.log&
        sleep 30
done
