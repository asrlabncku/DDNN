import chainer
import numpy as np

trainset, testset = chainer.datasets.get_cifar10(ndim=3)

#textfile = open("Output.txt","w")
#textfile.write(trainset[0][0])

#textfile.close()


#trainset[0][0].tofile("cifar10_train_1.txt",sep=",",format="%s")
#trainset[1][0].tofile("cifar10_train_2.txt",sep=",",format="%s")
#trainset[2][0].tofile("cifar10_train_3.txt",sep=",",format="%s")
#trainset[3][0].tofile("cifar10_train_4.txt",sep=",",format="%s")
#trainset[4][0].tofile("cifar10_train_5.txt",sep=",",format="%s")
#trainset[5][0].tofile("cifar10_train_6.txt",sep=",",format="%s")
#trainset[6][0].tofile("cifar10_train_7.txt",sep=",",format="%s")
#trainset[7][0].tofile("cifar10_train_8.txt",sep=",",format="%s")
#trainset[8][0].tofile("cifar10_train_9.txt",sep=",",format="%s")
#trainset[9][0].tofile("cifar10_train_10.txt",sep=",",format="%s")
#answer = ""
Ans = []
for i in range(10000):
    filename = "dataset/cifar10_test_" + str(i) + ".txt"
    testset[i][0].tofile( filename , sep="," , format="%s" )
    Ans.append(testset[i][1])

f = open('answer.txt','w')
for ans in Ans:
  print >> f, ans

#Ans.tofile( "answer.txt" , sep='\n' , format="%s" )
#f = open('answer.txt','w')
#f.write(answer)

#testset[0][0].tofile("cifar10_test_1.txt",sep=",",format="%s")
#testset[1][0].tofile("cifar10_test_2.txt",sep=",",format="%s")
#testset[2][0].tofile("cifar10_test_3.txt",sep=",",format="%s")
#testset[3][0].tofile("cifar10_test_4.txt",sep=",",format="%s")
#testset[4][0].tofile("cifar10_test_5.txt",sep=",",format="%s")
#testset[5][0].tofile("cifar10_test_6.txt",sep=",",format="%s")
#testset[6][0].tofile("cifar10_test_7.txt",sep=",",format="%s")
#testset[7][0].tofile("cifar10_test_8.txt",sep=",",format="%s")
#testset[8][0].tofile("cifar10_test_9.txt",sep=",",format="%s")
#testset[9][0].tofile("cifar10_test_10.txt",sep=",",format="%s")
#testset[10][0].tofile("cifar10_test_11.txt",sep=",",format="%s")
#testset[11][0].tofile("cifar10_test_12.txt",sep=",",format="%s")
#testset[12][0].tofile("cifar10_test_13.txt",sep=",",format="%s")
#testset[13][0].tofile("cifar10_test_14.txt",sep=",",format="%s")
#testset[14][0].tofile("cifar10_test_15.txt",sep=",",format="%s")
#testset[15][0].tofile("cifar10_test_16.txt",sep=",",format="%s")
#testset[16][0].tofile("cifar10_test_17.txt",sep=",",format="%s")
#testset[17][0].tofile("cifar10_test_18.txt",sep=",",format="%s")
#testset[18][0].tofile("cifar10_test_19.txt",sep=",",format="%s")
#testset[19][0].tofile("cifar10_test_20.txt",sep=",",format="%s")

#x = np.array([[[1.01, 2.232, 3.2214], [124.24,124.421,32.12]],[[0.01, 12.232, 3.2214], [124.24,124.421,32.12]]], np.float32)

#print(type(x))
#print(x.shape)
#print(x.dtype)

#print(type(trainset[0][0]))
#print(trainset[0][0].shape)

#print(trainset[0][0].dtype)

#print(len(trainset))

#for i in range(20):
#    print(testset[i][1])
#for i in range(len(trainset[0])):
#    print(type(trainset[0][i]));




