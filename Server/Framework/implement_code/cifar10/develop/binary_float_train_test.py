import os
from chainer_sequential.chain import Chain
import chainer
import chainer.serializers as S
from deepopt.trainer_fix import Trainer
from chainer_sequential.sequential import Sequential
from chainer_sequential.function import *
from chainer_sequential.link import *
from chainer_sequential.binary_link import *
from chainer import functions as F
from chainer import training
from chainer.training import extensions
from chainer.training.triggers import IntervalTrigger



folder = "models/binary_float"

nfilters_embeded = int(64)
nlayers_embeded = int(2)
nfilters_cloud = int(64)
nlayers_cloud = int(2)
branchweight = numpy.float64(0.1)
ent_T = numpy.int64(10)
lr = numpy.float64(0.001)
nepochs = int(80)
name = str("cifar10_64f_1t")

input_dims = 3
output_dims = 10


model = Sequential()


for i in range(nlayers_embeded):
    if i == 0:
        nfilters = input_dims
        model.add(ConvPoolBNBST(nfilters, nfilters_embeded, 3, 1, 1, 3, 1, 1))
    else:
        nfilters = nfilters_embeded
        model.add(BinaryConvPoolBNBST(nfilters, nfilters_embeded, 3, 1, 1, 3, 1, 1))

branch = Sequential()
branch.add(BinaryLinearBNSoftmax(None, output_dims))
model.add(branch)

# float branch
for i in range(nlayers_cloud):
    if i == 0:
        nfilters = nfilters_embeded
    else:
        nfilters = nfilters_cloud
    model.add(Convolution2D(nfilters, nfilters_cloud, 3, 1, 1))
    model.add(max_pooling_2d(3,1,1))
    model.add(BatchNormalization(nfilters_cloud))
    model.add(Activation('bst'))
    # Note: should we move pool to before batch norm like in binary?
model.add(Linear(None, output_dims))
model.add(BatchNormalization(output_dims))
        
model.build()


chain = Chain(branchweight=branchweight, ent_T=ent_T)
chain.add_sequence(model)
chain.setup_optimizers('adam', lr)

print("Model define Over ! ")

#################################################Training####################################################


print("Training Start !")

trainset, testset = chainer.datasets.get_cifar10(ndim=3)
trainer = Trainer('{}/{}'.format(folder,name), chain, trainset, testset, nepoch=nepochs, resume=True)
trainer.run()

print("Training Over !")

#trainer.load_model()
#################################################Inference ##################################################


#print("Inference  Start ! ")

#trainset, testset = chainer.datasets.get_cifar10(ndim=3)
#test_iter = chainer.iterators.SerialIterator(testset, 1,repeat=False, shuffle=False)
#chain.train = False
#chain.test = True
#chain.to_gpu(0)
#result = extensions.Evaluator(test_iter, chain, device=0)()

#print("Inference Over ! ")

#for key, value in result.iteritems() :
#    print key, value


#####################################################Generate Code############################################

print("Gen Code Start ! ")

save_file = "cifar10_inference_64f1t.h"
in_shape = (3,32,32)
c_code = model.generate_c(in_shape)
save_dir = os.path.join(os.path.split(save_file)[:-1])[0]
if not os.path.exists(save_dir) and save_dir != '':
    os.makedirs(save_dir)

with open(save_file, 'w+') as fp:
    fp.write(c_code)

print("Gen Code Over ! ")


