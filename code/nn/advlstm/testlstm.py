from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import numpy as np
import os

root="/Users/adminadmin/Documents/mywork/master/"
savepath=root+"code/nn/lstmloss2.txt"
trainF=root+"AVEC2014_AudioVisual/Label2/Training/Freeform/"
trainN=root+"AVEC2014_AudioVisual/Label2/Training/Northwind/"
devF=root+"AVEC2014_AudioVisual/Label2/Development/Freeform/"
devN=root+"AVEC2014_AudioVisual/Label2/Development/Northwind/"
testF=root+"AVEC2014_AudioVisual/Label2/Testing/Freeform/"
testN=root+"AVEC2014_AudioVisual/Label2/Testing/Northwind/"
def howmany():
    Ntrain=0
    Ndev=0
    Ntest=0
    for file in os.listdir(trainF):
        item=file.split(".")
        if (item[1]=="txt"):
            Ntrain+=1
    for file in os.listdir(trainN):
        item=file.split(".")
        if (item[1]=="txt"):
            Ntrain+=1

    for file in os.listdir(devF):
        item=file.split(".")
        if (item[1]=="txt"):
            Ndev+=1

    for file in os.listdir(devN):
        item=file.split(".")
        if (item[1]=="txt"):
            Ndev+=1

    for file in os.listdir(testF):
        item=file.split(".")
        if (item[1]=="txt"):
            Ntest+=1

    for file in os.listdir(testN):
        item=file.split(".")
        if (item[1]=="txt"):
            Ntest+=1
    return Ntrain,Ndev,Ntest

Ntrain,Ndev,Ntest=howmany()
total_train=Ntrain+Ndev

def findmaxlength():
    maxlength=0
    for file in os.listdir(trainF):
        item=file.split(".")
        if (item[1]=="txt"):
            txtpath=trainF+file
            m=np.loadtxt(txtpath)
            if (m.shape[0]>maxlength):
                maxlength=m.shape[0]
    for file in os.listdir(trainN):
        item=file.split(".")
        if (item[1]=="txt"):
            txtpath = trainN + file
            m = np.loadtxt(txtpath)
            if (m.shape[0] > maxlength):
                maxlength = m.shape[0]

    for file in os.listdir(devF):
        item=file.split(".")
        if (item[1]=="txt"):
            txtpath = devF + file
            m = np.loadtxt(txtpath)
            if (m.shape[0] > maxlength):
                maxlength = m.shape[0]

    for file in os.listdir(devN):
        item=file.split(".")
        if (item[1]=="txt"):
            txtpath = devN + file
            m = np.loadtxt(txtpath)
            if (m.shape[0] > maxlength):
                maxlength = m.shape[0]

    for file in os.listdir(testF):
        item=file.split(".")
        if (item[1]=="txt"):
            txtpath = testF + file
            m = np.loadtxt(txtpath)
            if (m.shape[0] > maxlength):
                maxlength = m.shape[0]

    for file in os.listdir(testN):
        item=file.split(".")
        if (item[1]=="txt"):
            txtpath = testN + file
            m = np.loadtxt(txtpath)
            if (m.shape[0] > maxlength):
                maxlength = m.shape[0]
    return maxlength
maxlength=findmaxlength()
print (maxlength)

def generate_train():
    train_set = np.zeros((total_train, maxlength, 3), dtype=np.float)
    train_label=np.zeros((total_train, 1),dtype=np.float)
    i = 0
    for file in os.listdir(trainF):
        item = file.split(".")
        if (item[1] == "txt"):
            txtpath = trainF + file
            m = np.loadtxt(txtpath, dtype=np.float)
            pad_area = m.shape[0]
            train_set[i][maxlength - pad_area:][:] = m
            name=item[0]
            namearray=name.split("_")
            dep=namearray[2]
            train_label[i][:]=dep
            i += 1

    for file in os.listdir(trainN):
        item = file.split(".")
        if (item[1] == "txt"):
            txtpath = trainN + file
            m = np.loadtxt(txtpath, dtype=np.float)
            pad_area = m.shape[0]
            train_set[i][maxlength - pad_area:][:] = m
            name = item[0]
            namearray = name.split("_")
            dep = namearray[2]
            train_label[i][:] = dep
            i += 1

    for file in os.listdir(devF):
        item = file.split(".")
        if (item[1] == "txt"):
            txtpath = devF + file
            m = np.loadtxt(txtpath, dtype=np.float)
            pad_area = m.shape[0]
            train_set[i][maxlength - pad_area:][:] = m
            name = item[0]
            namearray = name.split("_")
            dep = namearray[2]
            train_label[i][:] = dep
            i += 1

    for file in os.listdir(devN):
        item = file.split(".")
        if (item[1] == "txt"):
            txtpath = devN + file
            m = np.loadtxt(txtpath, dtype=np.float)
            pad_area = m.shape[0]
            train_set[i][maxlength - pad_area:][:] = m
            name = item[0]
            namearray = name.split("_")
            dep = namearray[2]
            train_label[i][:] = dep
            i += 1
            if (i == total_train):
                break
    return train_set,train_label

def generate_val():
    total_val=int(Ntest/2)

    val_set = np.zeros((total_val, maxlength, 3), dtype=np.float) #203-249
    val_label = np.zeros((total_val, 1), dtype=np.float) #250-368

    test_set = np.zeros((total_val, maxlength, 3), dtype=np.float)
    test_label=np.zeros((total_val, 1),dtype=np.float)
    i = 0
    j=0
    for file in os.listdir(testF):
        item = file.split(".")
        if (item[1] == "txt"):
            header=int(file.split("_")[0])
            txtpath = testF + file
            m = np.loadtxt(txtpath, dtype=np.float)
            pad_area = m.shape[0]
            name = item[0]
            namearray = name.split("_")
            dep = namearray[2]
            if (header<250):
                val_set[i][maxlength - pad_area:][:] = m
                val_label[i][:] = dep
                i += 1
            else:
                test_set[j][maxlength - pad_area:][:] = m
                test_label[j][:] = dep
                j += 1

    for file in os.listdir(testN):
        item = file.split(".")
        if (item[1] == "txt"):
            header = int(file.split("_")[0])
            txtpath = testN + file
            m = np.loadtxt(txtpath, dtype=np.float)
            pad_area = m.shape[0]
            name = item[0]
            namearray = name.split("_")
            dep = namearray[2]
            if (header<250):
                val_set[i][maxlength - pad_area:][:] = m
                val_label[i][:] = dep
                i += 1
            else:
                test_set[j][maxlength - pad_area:][:] = m
                test_label[j][:] = dep
                j += 1
            if (i == total_train and j == total_train):
                break
    return val_set, val_label, test_set, test_label

X_train,Y_train=generate_train()     #X_train(200, 7440, 3) Y_train(200, 1)
X_val,Y_val, X_test,Y_test=generate_val()

def accuracy(Y_pre,Y_test):
    Y_pre1=Y_pre.flatten()
    Y_test1=Y_test.flatten()
    N=len(Y_pre1)
    Y_pre2=np.zeros(N,np.int)
    Y_test2=np.zeros(N,np.int)
    Y_pre3 = np.zeros(N, np.int)
    Y_test3 = np.zeros(N, np.int)

    for i in range(N):
        if (Y_pre1[i]<14):
            Y_pre2[i]=0
        elif(Y_pre1[i]>=14 and Y_pre1[i]<20):
            Y_pre2[i] = 1
        elif (Y_pre1[i] >= 20 and Y_pre1[i] < 29):
            Y_pre2[i] = 2
        elif (Y_pre1[i] >= 29 and Y_pre1[i] <= 63):
            Y_pre2[i] = 3

    for i in range(N):
        if (Y_test1[i]<14):
            Y_test2[i]=0
        elif(Y_test1[i]>=14 and Y_test1[i]<20):
            Y_test2[i] = 1
        elif (Y_test1[i] >= 20 and Y_test1[i] < 29):
            Y_test2[i] = 2
        elif (Y_test1[i] >= 29 and Y_test1[i] <= 63):
            Y_test2[i] = 3


    correct_number = float((Y_pre2 == Y_test2).sum())
    accuracy = correct_number / N

    #binary classification
    for i in range(N):
        if (Y_pre1[i]>=0 and Y_pre1[i]<14):
            Y_pre3[i]=0
        else:
            Y_pre3[i] = 1

    for i in range(N):
        if (Y_test1[i]>=0 and Y_test1[i]<14):
            Y_test3[i]=0
        else:
            Y_test3[i] = 1

    correct_number1 = float((Y_pre3 == Y_test3).sum())
    accuracy1 = correct_number1 / N
    #1st confusion matrix
    confusion_matrix=np.zeros((4,4),dtype=np.int) # order by 0,1,2,3

    for index1 in range(4):
        for index2 in range(4):
            for index3 in range(N):
                if (Y_test2[index3] == index1 and Y_pre2[index3] == index2):
                    confusion_matrix[index1][index2] += 1

    confusion_matrix1 = np.zeros((2, 2), dtype=np.int)  # order by 0,1

    for index1 in range(2):
        for index2 in range(2):
            for index3 in range(N):
                if (Y_test3[index3] == index1 and Y_pre3[index3] == index2):
                    confusion_matrix1[index1][index2] += 1

    return accuracy,accuracy1,confusion_matrix,confusion_matrix1
BS=30
EP=2
model = keras.models.load_model(root+"code/nn/weights_685.hdf5")
Y_pre=model.predict(X_test)
accuracy,accuracy1,confusion_matrix,confusion_matrix1=accuracy(Y_pre,Y_test)
print (accuracy)
print (confusion_matrix)
print("========")
print (accuracy1)
print (confusion_matrix1)