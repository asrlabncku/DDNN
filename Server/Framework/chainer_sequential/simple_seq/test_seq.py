from chainer_sequential.chain import Chain
from deepopt.trainer import Trainer
import chainer
import chainer.serializers as S
from chainer_sequential.sequential import Sequential
from chainer_sequential.function import *
from chainer_sequential.link import *
from chainer_sequential.binary_link import *
from chainer import functions as F

folder = "_models/test_seq"

nfilters_embeded = int(2)
nlayers_embeded = int(1)
branchweight = int(3)
lr = numpy.float64(0.001)
nepochs = int(2)
name = str("_lr_0.001_nfilters_embeded_2_nlayers_embeded_1.0")

model = Sequential()

nfilters = 1
model.add(ConvPoolBNBST(nfilters, nfilters_embeded, 3, 1, 1, 3, 1, 1))
nfilters = nfilters_embeded
model.add(BinaryConvPoolBNBST(nfilters, nfilters_embeded, 3, 1, 1, 3, 1, 1))
model.add(BinaryLinearBNSoftmax(None, 10))
model.build()

chain = Chain(branchweight =branchweight )
chain.add_sequence(model)
chain.setup_optimizers('adam', lr)

trainset, testset = chainer.datasets.get_mnist(ndim=3)

trainer = Trainer('{}/{}'.format(folder,name), chain, trainset, testset, nepoch=nepochs, resume=True)
trainer.run()

#for i in xrange(100):
#	y = chain(x)
#	loss = F.mean_squared_error(x, y)
#	chain.backprop(loss)
#	print float(loss.data)
#
#chain.save("model")

#y = Variable(chain)
#y = chain(x)

