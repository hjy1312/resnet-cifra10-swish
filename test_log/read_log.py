import re
import numpy as np
import matplotlib.pyplot as plt
log_prefix = 'test'
log_postfix = '.log'
best_acc = 0.0
best_index = 1
acc_set = []
model_set = []
for i in xrange(50):
    i += 1
    log = log_prefix+str(int(i))+log_postfix
    with open(log,'r') as f:
        lines = f.readlines()
        last_line = lines[-1]
        last_line = last_line.split()
        acc = float(last_line[-1])
    f.close()
    if(acc>best_acc):
        best_acc = acc
        best_index = i
    acc_set = acc_set + [acc]
    fp = open(log,'r')
    while 1:
	line = fp.readline()
	if not line:
	    break
	s = re.findall(r'_iter_(\w+).caffemodel',line,re.M)
	if s !=[]:
            model_set = model_set + [int(s[0])]
    fp.close()
print 'best acc is',best_acc
print 'in the log:',best_index
acc_set = np.array(acc_set)
model_set = np.array(model_set)
index_set = np.argsort(model_set)
model_set_sorted = model_set[index_set]
acc_set = acc_set[index_set]
plt.plot(model_set_sorted,acc_set)
plt.xlabel('model')
plt.ylabel('accuracy')
plt.show()
