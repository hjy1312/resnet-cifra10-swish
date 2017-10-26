resnet using swish activation and cifar 10,it achieve about 0.25% higher accuracy than the relu version.

test.py is used to test the caffemodel and get the best accuracy among so many models,get_test_data.py is used to get cifar10 test data.

all_test.sh is used to test the caffemodel one by one and write the results into log files,then run read_log.py(in test_log folder) to get the best accuracy and plot the accuracy curve.

