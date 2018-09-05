
from chainer_sequential.chain import Chain
from deepopt.trainer import Trainer
import chainer
import chainer.serializers as S
from chainer_sequential.sequential import Sequential
from chainer_sequential.function import *
from chainer_sequential.link import *
from chainer_sequential.binary_link import *
from chainer import functions as F
#import chainer.computational_graph as c
from chainer import computational_graph

folder = "_models/test_seq"

nfilters_embeded = int(2)
nlayers_embeded = int(1)
nfilters_cloud = int(2)
nlayers_cloud = int(1)
branchweight = int(3)
lr = numpy.float64(0.001)
nepochs = int(10)
name = str("_lr_0.001_nfilters_embeded_2_nlayers_embeded_1.0")

model = Sequential()

nfilters = 1
model.add(ConvPoolBNBST(nfilters, nfilters_embeded, 3, 1, 1, 3, 1, 1))
#model.add(ConvBNBST(nfilters, nfilters_embeded, 3, 1, 1))
nfilters = nfilters_embeded
model.add(BinaryConvPoolBNBST(nfilters, nfilters_embeded, 3, 1, 1, 3, 1, 1))
#model.add(BinaryConvBNBST(nfilters, nfilters_embeded, 3, 1, 1))


branch = Sequential()
branch.add(BinaryLinearBNSoftmax(None, 10))
model.add(branch)

############################ Embeded network ##############################

nfilters = nfilters_embeded
model.add(Convolution2D(nfilters, nfilters_embeded, 3, 1, 1))
model.add(BatchNormalization(nfilters_embeded))
model.add(Activation('relu'))
model.add(max_pooling_2d(3,1,1))

############################# Cloud network ###############################

nfilters = nfilters_cloud
model.add(Convolution2D(nfilters, nfilters_cloud, 3, 1, 1))
model.add(BatchNormalization(nfilters_cloud))
model.add(Activation('relu'))
model.add(max_pooling_2d(3,1,1))



model.add(Linear(None, 10))
model.build()


#model.add(BinaryLinearBNSoftmax(None, 10))
#model.build()

chain = Chain(branchweight =branchweight )
chain.add_sequence(model)
chain.setup_optimizers('adam', lr)

#for i in xrange(100):
#	y = chain(x)
#	loss = F.mean_squared_error(x, y)
#	chain.backprop(loss)
#	print float(loss.data)
#
#chain.save("model")


trainset, testset = chainer.datasets.get_mnist(ndim=3)

trainer = Trainer('{}/{}'.format(folder,name), chain, trainset, testset, nepoch=nepochs, resume=True)
trainer.run()

#y = Variable(chain)
#y = chain(x)


#g = c.build_computational_graph(chain)
#with open('graph.dot', 'w') as o:
#    o.write(g.dump())

#with open('graph.dot', 'w') as o:
#        g = computational_graph.build_computational_graph((y, ))
#        o.write(g.dump())






