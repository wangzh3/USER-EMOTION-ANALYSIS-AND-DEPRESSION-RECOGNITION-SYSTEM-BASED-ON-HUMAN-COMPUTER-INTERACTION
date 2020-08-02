import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
def tocategory(Y_pre1,num_classes):
    N=len(Y_pre1)
    Y_pre2 = np.zeros(N, np.int)
    if(num_classes==4):
        for i in range(N):
            if (Y_pre1[i] < 14):
                Y_pre2[i] = 0
            elif (Y_pre1[i] >= 14 and Y_pre1[i] < 20):
                Y_pre2[i] = 1
            elif (Y_pre1[i] >= 20 and Y_pre1[i] < 29):
                Y_pre2[i] = 2
            elif (Y_pre1[i] >= 29 and Y_pre1[i] <= 63):
                Y_pre2[i] = 3
        return Y_pre2
    elif (num_classes==2):
        for i in range(N):
            if (Y_pre1[i] < 14):
                Y_pre2[i] = 0
            else:
                Y_pre2[i] = 1
        return Y_pre2

def confusion(Y_test,Y_pre,num_classes):
    N=len(Y_test)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int)  # order by 0,1,2,3
    for index1 in range(num_classes):
        for index2 in range(num_classes):
            for index3 in range(N):
                if (Y_test[index3] == index1 and Y_pre[index3] == index2):
                    confusion_matrix[index1][index2] += 1
    return confusion_matrix

relationdata=pd.read_csv("relation.csv")
relation=relationdata.values
dep=relation[:,9]

traindata=[]
traindep=[]

bad=np.array([0,0,0,0,0,0,1],dtype=np.float)

for i in range(relation.shape[0]):
    if (relation[i, 2:9]==bad).all()==False:
        traindata.append(relation[i, 2:9].tolist())
        traindep.append(dep[i])
#change dep to category value
traindep1=tocategory(traindep,4)

testcsv=pd.read_csv("test.csv")
testnp=testcsv.values
dep1=testnp[:,9]

valdata=[]
valdep=[]

testdata=[]
testdep=[]

for i in range(testnp.shape[0]):
    header=int(testnp[i][0].split("_")[0])
    if (header<250):
        valdata.append(testnp[i, 2:9].tolist())
        valdep.append(dep1[i])
    else:
        testdata.append(testnp[i, 2:9].tolist())
        testdep.append(dep1[i])

valdep1 = tocategory(valdep,4)

#find the best number of neighbors for 4 classes
X=[]
A=[]
for i in range(1,51):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(traindata, traindep1)
    predict = neigh.predict(valdata)
    acc = sum(valdep1 == predict) / len(valdep1)
    print(str(i)+" "+str(acc))
    X.append(i)
    A.append(acc)

#test for 4 classes
testdep1=tocategory(testdep,4)
print ("test 4 classes:")
print ("original test class:")
confusion_matrix1=confusion(testdep1,testdep1,4)
print (confusion_matrix1)

print ("predict test class:")
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(traindata, traindep1)
testpredict = neigh.predict(testdata)

acc = sum(testdep1 == testpredict) / len(testdep1)
print(acc)

confusion_matrix=confusion(testdep1,testpredict,4)
print (confusion_matrix)

#test for 2 classes
testdep2=tocategory(testdep,2)
print ("test 2 classes:")
print ("original test class:")
confusion_matrix1=confusion(testdep2,testdep2,2)
print (confusion_matrix1)

print ("predict test class k=5:")
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(traindata, traindep1)
testpredict = neigh.predict(testdata)
testpredict2=np.where(testpredict>0,1,0)

acc = sum(testdep2 == testpredict2) / len(testdep2)
print(acc)

confusion_matrix=confusion(testdep2,testpredict2,2)
print (confusion_matrix)

print ("predict test class k=27:")
neigh = KNeighborsClassifier(n_neighbors=27)
neigh.fit(traindata, traindep1)
testpredict = neigh.predict(testdata)
testpredict2=np.where(testpredict>0,1,0)

acc = sum(testdep2 == testpredict2) / len(testdep2)
print(acc)

confusion_matrix=confusion(testdep2,testpredict2,2)
print (confusion_matrix)


print("average method:")
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

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(vecmatrix, [0,1,2,3])
testpredict = neigh.predict(testdata)

acc = sum(testdep1 == testpredict) / len(testdep1)
print(acc)

confusion_matrix=confusion(testdep1,testpredict,4)
print (confusion_matrix)

testpredict2=np.where(testpredict>0,1,0)

acc = sum(testdep2 == testpredict2) / len(testdep2)
print(acc)

confusion_matrix=confusion(testdep2,testpredict2,2)
print (confusion_matrix)

plt.figure()
plt.plot(X,A,color='black')
plt.xlabel('K value')
plt.ylabel('val accuracy')
plt.show()