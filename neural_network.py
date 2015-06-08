__author__ = 'mong'
from network import *
from load_data import *
import cPickle as pickle
from time import time
import numpy as np
import pybrain
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.structure.modules   import SoftmaxLayer
#from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
#from scipy import diag, arange, meshgrid, where
#from numpy.random import multivariate_normal
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader

print "Training With Neural network"
print("Loading training and test labels")
[trainData, trainLabel] = loadCifar(10)
del trainData

Name = 'trainLabel.txt'
np.savetxt(Name, np.array(trainLabel))


[testData, testLabel] = loadCifar(6)
del testData

Name = 'testLabel.txt'
np.savetxt(Name, np.array(testLabel))


# Load Training and Test Data/Extracted from DeSTIN

# here we do not use the whole set of feature extracted from DeSTIN
# We use the features which are extracted from the top few layers
print("Loading training and testing features")


I = 199
Name = 'p_train/' + str(I + 1) + '.txt'
trainData = np.ravel(np.loadtxt(Name))


for I in range(399, 50000, 200):
    Name = 'p_train/' + str(I + 1) + '.txt'
    file_id = open(Name, 'r')
    Temp = np.ravel(np.loadtxt(Name))
    trainData = np.hstack((trainData, Temp))

del Temp

Name = 'trainData.txt'
np.savetxt(Name, np.array(trainData))


Len = np.shape(trainData)[0]
Size = np.size(trainData)
trainLabel = np.squeeze(np.asarray(trainLabel).reshape(50000, 1))



I = 199
Name = 'p_test/' + str(I + 1) + '.txt'
testData = np.ravel(np.loadtxt(Name))

for I in range(399, 10000, 200):
    Name = 'p_test/' + str(I + 1) + '.txt'
    file_id = open(Name, 'r')
    Temp = np.ravel(np.loadtxt(Name))
    testData = np.hstack((testData, Temp))

del Temp

Len = np.shape(testData)[0]
Size = np.size(testData)
dim = Len/50000

testData = np.array([])
print("Loading training and testing features")


for I in range(199, 10000, 200):
    Name = 'p_test/' + str(I + 1) + '.txt'
    file_id = open(Name, 'r')
    Temp = np.ravel(np.loadtxt(Name))
    testData = np.hstack((testData, Temp))
del Temp

Name = 'testData.txt'
np.savetxt(Name, np.array(testData))

Len = np.shape(testData)[0]
Size = np.size(testData)



deta=50000

Name = open('trainData.txt', 'r')
trainData = np.ravel(np.loadtxt(Name))

Len = np.shape(trainData)[0]

#load train data
train = trainData.reshape( 50000, dim)
x_train = train[0:deta,:] 
Name = open('trainLabel.txt', 'r')

trainLabel = np.ravel(np.loadtxt(Name))
Label = trainLabel.reshape( 50000, 1 )
y_train = Label[0:deta,:]  
#################

#load test data
Name = open('testData.txt', 'r')
testData = np.ravel(np.loadtxt(Name))

Len = np.shape(testData)[0]

x_test = testData.reshape( 10000, dim)

Name = open('testLabel.txt', 'r')

testLabel = np.ravel(np.loadtxt(Name))
y_test = testLabel.reshape( 10000, 1 )
###################
#initialize network
DS = pybrain.datasets.classification.ClassificationDataSet(inp=dim, nb_classes=10, class_labels=['airplane' , 'automobile', 'bird','cat' ,'deer','dog','frog','horse','ship','truck'])


for i in range(deta): 
   DS.appendLinked(x_train[i]   , [int(y_train[i])])

DS._convertToOneOfMany(bounds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
net = buildNetwork(, 50, 10, bias=True, outclass=SoftmaxLayer)
print "loading network"
net=NetworkReader.readFrom('data/net')
#####################

#train the model
trainer = BackpropTrainer(net, DS )
err=0
err_p=1
error = 10
iteration = 0
print 'start'
while error > 0.1 or err_p>.2:
  error = trainer.train()
  iteration += 1
  for i in range(deta): 
	   result = net.activate(x_train[i])
           result = result.argmax(axis=0)
           if result != int(y_train[i]):
              err=err+1
  err_p=err/float(deta)
  err=0

#test the model
  test_err=0
  for i in range(10000): 
    test_result = net.activate(x_test[i])
    test_result = test_result.argmax(axis=0)
    if test_result != int(y_test[i]):
       test_err=test_err+1
  test_err_p=test_err/float(10000)
  test_err=0

#print result      
  print "Iteration:{0} Error: {1} train_accuracy: {2} test_acc: {3}".format(iteration, error, 1-err_p, 1-test_err_p)
  if iteration%50==0:
     print 'save'
     NetworkWriter.writeToFile(net,'data/net')
  if iteration%200==0:
     NetworkWriter.writeToFile(net,'data/net'+str(iteration))
NetworkWriter.writeToFile(net,'net')


