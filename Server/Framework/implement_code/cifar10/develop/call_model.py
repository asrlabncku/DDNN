import os
import sys
sys.path.append('..')

import chainer

from elaas.family.cifar10_model_4f3l import Cifar10_4f3l


a = Cifar10_4f3l()()

print 'result = %s' % a


