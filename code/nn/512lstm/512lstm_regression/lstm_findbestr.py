import numpy as np
from matplotlib import pyplot as plt
matrix=np.loadtxt("/Users/adminadmin/Documents/mywork/master/code/nn/512lstm/lstmloss.txt",dtype=np.str)
data=matrix[1:,:].astype(np.float)
data1=data[:,1]
index=np.where(data1==np.min(data1))[0][0]
epoch=data[index][0]
loss=data[index][1]
print (epoch)
print (loss)

data0=data[:,0]
plt.figure()
plt.plot(data0,data1,color='black')
plt.xlabel('Epoch')
plt.ylabel('validation loss')
plt.show()