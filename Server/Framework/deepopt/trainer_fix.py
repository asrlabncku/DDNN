from chainer import training
from chainer.training import extensions
from chainer.optimizers import adam as A
import chainer.serializers as S
import chainer
import os
import json
from chainer.training.triggers import IntervalTrigger
from collections import defaultdict
import numpy as np

class Trainer(object):
    def __init__(self, folder, chain, train, test, batchsize=500, resume=True, gpu=0, nepoch=1, reports=[]):
        self.reports = reports
        self.nepoch = nepoch
        self.folder = folder
        self.chain = chain
        self.gpu = gpu
        
        if self.gpu >= 0:
            chainer.cuda.get_device(gpu).use()
            chain.to_gpu(gpu)
        self.eval_chain = eval_chain = chain.copy()
        self.chain.test = False
        self.eval_chain.test = True
        self.testset = test

        if not os.path.exists(folder):
            os.makedirs(folder)

        train_iter = chainer.iterators.SerialIterator(train, batchsize, shuffle=True)
        test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                     repeat=False, shuffle=False)
        updater = training.StandardUpdater(train_iter, chain.optimizer, device=gpu)
        trainer = training.Trainer(updater, (nepoch, 'epoch'), out=folder)
        # trainer.extend(TrainingModeSwitch(chain))
        #trainer.extend(extensions.dump_graph('main/loss'))
        trainer.extend(extensions.Evaluator(test_iter, eval_chain, device=gpu), trigger=(1,'epoch'))
        trainer.extend(extensions.snapshot_object(
            chain, 'so_epoch_{.updater.epoch:06}'), trigger=(1,'epoch'))
        trainer.extend(extensions.snapshot(
            filename='s_epoch_{.updater.epoch:06}'), trigger=(1,'epoch'))
        trainer.extend(extensions.LogReport(trigger=(1,'epoch')), trigger=(1,'iteration'))
        trainer.extend(extensions.PrintReport(
            ['epoch']+reports), trigger=IntervalTrigger(1,'epoch'))

        self.trainer = trainer
        
        if resume:
            #if resumeFrom is not None:
            #    trainerFile = os.path.join(resumeFrom[0],'snapshot_epoch_{:06}'.format(resumeFrom[1]))
            #    S.load_npz(trainerFile, trainer)
            #print("test enter trainer __init__ resume")
            i = 1
            trainerFile = os.path.join(folder,'snapshot_epoch_{:06}'.format(i))
            while i <= nepoch and os.path.isfile(trainerFile):
                i = i + 1
                trainerFile = os.path.join(folder,'snapshot_epoch_{:06}'.format(i))
            i = i - 1
            trainerFile = os.path.join(folder,'snapshot_epoch_{:06}'.format(i))
            if i >= 0 and os.path.isfile(trainerFile):
                S.load_npz(trainerFile, trainer)


    def load_model(self):
        batchsize = 500
        train, test = chainer.datasets.get_mnist(ndim=3)
        optimizer = A.Adam()
        #print(type(optimizer))
        train_iter = chainer.iterators.SerialIterator(train, batchsize, shuffle=True)
        updater = training.StandardUpdater(train_iter, optimizer, device=0)
        
        folder = self.folder
        trainerFile = os.path.join(folder,'snapshot_epoch_{:06}'.format(1))
        print(trainerFile)
        model = training.Trainer(updater, stop_trigger=None, out=folder)
        S.load_npz(trainerFile, model)
        print("test load_model() over !")


    def run(self):
        print("test run start ! ")
        if self.gpu >= 0:
            chainer.cuda.get_device(self.gpu).use()
            self.chain.to_gpu(self.gpu)
        self.chain.test = False
        self.eval_chain.test = True
        print("test run process 1")    
        self.trainer.run()
        print("test run process 2")
        #ext = self.trainer.get_extension('validation')()
        #test_accuracy = ext['validation/main/accuracy']
        #test_loss = ext['validation/main/loss']
        #acc = test_accuracy.tolist()
        #loss = test_loss.tolist()
        if self.gpu >= 0:
            self.chain.to_cpu()
        print("test run process 3")
        #return self.evaluate()
        #return acc,loss

    def evaluate(self):
        print("test evaluate start ! ")
        test_iter = chainer.iterators.SerialIterator(self.testset, 1,
                                                     repeat=False, shuffle=False)
        print("test evaluate process 1")
        self.chain.train = False
        self.chain.test = True
        if self.gpu >= 0:
            self.chain.to_gpu(self.gpu)
        print("test evaluate process 2")
        result = extensions.Evaluator(test_iter, self.chain, device=self.gpu)()
        print("test evaluate process 3")
        if self.gpu >= 0:
            self.chain.to_cpu()
        #for k,v in result.iteritems():
        #    if k in ["main/numsamples", "main/accuracy", "main/branch0exit", "main/branch1exit", "main/branch2exit"]:
        #        print k, "\t\t\t", v
        print("test evaluate provess 4")
        return result
    
    

    def save_model(self):
        trainer = self.trainer
        chain = self.chain
        trainer.extend(extensions.snapshot_object(chain, 'so_epoch_{.updater.epoch:06}'), trigger=(1,'epoch'))
        trainer.extend(extensions.snapshot(filename='s_epoch_{.updater.epoch:06}'), trigger=(1,'epoch'))


    # Deprecated
    def get_result(self, key):
        # this only returns the lastest log
        ext = self.trainer.get_extension('validation')()
        return ext.get('{}'.format(key),np.array(None)).tolist()

    def get_log_result(self, key, reduce_fn=np.mean):
        folder = self.folder
        nepoch = self.nepoch
        with open(os.path.join(folder,'log'),'r') as f:
            log = json.load(f)

        epochMap = defaultdict(list)
        for v in log:
            if v.get(key,None) is not None:
                epochMap[int(v["epoch"])].append(v[key])

        epochs = sorted(epochMap.keys())
        values = []
        for epoch in epochs:
            values.append(reduce_fn(epochMap[epoch]))

        return values[:nepoch]
