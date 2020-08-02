import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

relationdata=pd.read_csv("relation.csv")
relation=relationdata.values
dep=relation[:,9]

vecmatrix=np.zeros((4,7),dtype=np.float)
number=np.zeros(4,dtype=np.int)

bad=np.array([0,0,0,0,0,0,1],dtype=np.float)

for i in range(relation.shape[0]):
    if (relation[i, 2:9]==bad).all()==False:
        if (dep[i] < 14):
            number[0] += 1
            vecmatrix[0,:] = vecmatrix[0,:]+relation[i, 2:9]
        elif (dep[i] >= 14 and dep[i] < 20):
            number[1] += 1
            vecmatrix[1,:] = vecmatrix[1,:]+relation[i, 2:9]
        elif (dep[i] >= 20 and dep[i] < 29):
            number[2] += 1
            vecmatrix[2,:] = vecmatrix[2,:]+relation[i, 2:9]
        elif (dep[i] >= 29 and dep[i] <= 63):
            number[3] += 1
            vecmatrix[3,:] = vecmatrix[3,:]+relation[i, 2:9]

for i in range(4):
    vecmatrix[i] = vecmatrix[i] / number[i]

dist=np.zeros((4,4),dtype=np.float)

for i in range(4):
    for j in range(4):
        dist[i][j]=np.linalg.norm(vecmatrix[i] - vecmatrix[j])
print (number)
print (dist)
#binary

vecmatrix=np.zeros((2,7),dtype=np.float)
number=np.zeros(2,dtype=np.int)

for i in range(relation.shape[0]):
    if(relation[i, 2:9] == bad).all() == False:
        if (dep[i] < 14):
            number[0] += 1
            vecmatrix[0,:] = vecmatrix[0,:]+relation[i, 2:9]
        else:
            number[1] += 1
            vecmatrix[1,:] = vecmatrix[1,:]+relation[i, 2:9]

for i in range(2):
    vecmatrix[i] = vecmatrix[i] / number[i]

dist=np.linalg.norm(vecmatrix[1] - vecmatrix[0])
print (number)
print (dist)