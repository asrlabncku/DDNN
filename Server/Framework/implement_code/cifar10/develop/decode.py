import os
from chainer_sequential.chain_fix import Chain
import chainer
import chainer.serializers as S
from chainer_sequential.sequential_fix import Sequential
from chainer_sequential.function import *
from chainer_sequential.link import *
from chainer_sequential.binary_link import *
from chainer import functions as F
from chainer import training
from chainer.training import extensions
from chainer.training.triggers import IntervalTrigger

import numpy as np

data_path = "pass.txt"
string = open(data_path, 'r').read()
model , data = string.split("@")
data_string , layer_number = data.split(":")


k = data_string.split("n")
n = []
f = []
w = []

num_f = 0
num_w = 0
for p in k:
    f.append([])
    w.append([])
    n.append(p.split("f"))
    for q in p.split("f"):
        w[num_f].append([])
        f[num_f].append(q.split("|"))
        for o in q.split("|"):

            g = []
            s = ""
            for x in o.split(","):
                code = '{0:08b}'.format(int(x))
                s = s + code
            #print s
            for i in range(len(s)):
                if s[i] == "1" :
                    g.append("1")
                else : 
                    g.append("-1")
            w[num_f][num_w].append(g)
        num_w = num_w + 1

    num_f = num_f + 1
    num_w = 0

x = np.asarray(w,dtype=np.float32)
x_data = Variable(x, volatile='on')

print "data shape : %s , data type : %s" % (x_data.data.shape , x_data.data.dtype)
print "data : %s" % x_data.data



