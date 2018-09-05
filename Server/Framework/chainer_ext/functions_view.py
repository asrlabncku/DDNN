from chainer import function,cuda
import chainer.functions as F
class Entropy(function.Function):
    def forward(self, x):
        xp = cuda.get_array_module(*x)
        print "x[0] : %s" % (x[0])
        y = x[0] * xp.log(x[0]+1e-9)
        print "y : %s" % (y)
        print "sum_result : %s" % (-xp.sum(y,1))
        return -xp.sum(y,1),
        
    def backward(self, x, gy):
        return gy,
    
def entropy(x):
    return Entropy()(x)
