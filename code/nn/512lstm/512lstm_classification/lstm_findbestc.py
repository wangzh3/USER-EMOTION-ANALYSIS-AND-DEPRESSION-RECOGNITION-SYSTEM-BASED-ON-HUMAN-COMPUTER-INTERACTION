import numpy as np
from matplotlib import pyplot as plt
matrix=np.loadtxt("/Users/adminadmin/Documents/mywork/master/code/nn/512lstm_classification/lstmloss.txt")
valloss=matrix[:,0]
index=np.where(valloss==np.min(valloss))[0][0]
epoch=index+1
loss=valloss[index]
print ("loss")
print (epoch)
print (loss)

valacc=matrix[:,1]
index=np.where(valacc==np.max(valacc))[0][0]
epoch=index+1
acc=valacc[index]
print ("acc")
print (epoch)
print (acc)

data0=np.zeros(len(valloss),np.int)
for i in range(len(valloss)):
    data0[i]=i+1

plt.figure("val loss")
plt.plot(data0,valloss,color='black')
plt.xlabel('Epoch')
plt.ylabel('validation loss')

plt.figure("val accuracy")
plt.plot(data0,valacc,color='black')
plt.xlabel('Epoch')
plt.ylabel('validation accuracy')
plt.show()