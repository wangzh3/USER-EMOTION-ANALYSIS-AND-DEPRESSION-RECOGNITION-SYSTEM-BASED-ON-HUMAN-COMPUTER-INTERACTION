from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
import random
root="/Users/adminadmin/Documents/mywork/master/"
savepath=root+"code/nn/512lstm/lstmloss.txt"
trainF=root+"AVEC2014_AudioVisual/Label512/Training/Freeform/"
trainN=root+"AVEC2014_AudioVisual/Label512/Training/Northwind/"
devF=root+"AVEC2014_AudioVisual/Label512/Development/Freeform/"
devN=root+"AVEC2014_AudioVisual/Label512/Development/Northwind/"
testF=root+"AVEC2014_AudioVisual/Label512/Testing/Freeform/"
testN=root+"AVEC2014_AudioVisual/Label512/Testing/Northwind/"
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
print (howmany())

def num_lines(txtfile):
    N=0
    for line in open(txtfile):
        N+=1
    return N

def findmaxlength():
    maxlength=0
    for file in os.listdir(trainF):
        item=file.split(".")
        if (item[1]=="txt"):
            txtpath=trainF+file
            m=num_lines(txtpath)
            if (m>maxlength):
                maxlength=m
    for file in os.listdir(trainN):
        item=file.split(".")
        if (item[1]=="txt"):
            txtpath = trainN + file
            m = num_lines(txtpath)
            if (m> maxlength):
                maxlength = m

    for file in os.listdir(devF):
        item=file.split(".")
        if (item[1]=="txt"):
            txtpath = devF + file
            m = num_lines(txtpath)
            if (m > maxlength):
                maxlength = m

    for file in os.listdir(devN):
        item=file.split(".")
        if (item[1]=="txt"):
            txtpath = devN + file
            m = num_lines(txtpath)
            if (m > maxlength):
                maxlength = m

    for file in os.listdir(testF):
        item=file.split(".")
        if (item[1]=="txt"):
            txtpath = testF + file
            m = num_lines(txtpath)
            if (m > maxlength):
                maxlength = m

    for file in os.listdir(testN):
        item=file.split(".")
        if (item[1]=="txt"):
            txtpath = testN + file
            m = num_lines(txtpath)
            if (m > maxlength):
                maxlength = m
    return maxlength
maxlength=findmaxlength()
print (maxlength)

def generate1(testtxt):
    patharray=np.loadtxt(testtxt,dtype=np.str)
    while True:
        for pathname in patharray:
            filename=pathname.split("/")[-1]
            tail = filename.split(".")[1]
            head = filename.split(".")[0]
            if (tail == "txt"):
                X=np.loadtxt(pathname)
                X1=X.reshape(1, X.shape[0], 512)
                X_batch = np.zeros((1, maxlength, 512))
                pad_area = X.shape[0]
                X_batch[0][maxlength - pad_area:][:] = X1
                depvalue = int(head.split("_")[2])
                depvalue_array=np.zeros(1,dtype=np.int)
                depvalue_array[0]=depvalue
                yield (X_batch, depvalue_array)

def generatelabel(testtxt):
    patharray=np.loadtxt(testtxt,dtype=np.str)
    label=[]
    for pathname in patharray:
        filename = pathname.split("/")[-1]
        tail = filename.split(".")[1]
        head = filename.split(".")[0]
        if (tail == "txt"):
            depvalue = int(head.split("_")[2])
            label.append(depvalue)
    return np.array(label)


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
model = keras.models.load_model(root+"code/nn/512lstm_regression/lstmsave512/weights_1983.hdf5")
'''
X=np.loadtxt("/Users/adminadmin/Documents/mywork/master/AVEC2014_AudioVisual/Label512/Testing/Freeform/203_2_8.txt")
X1=X.reshape(1, X.shape[0], 512)
maxlength=7440
X_batch = np.zeros((1, maxlength, 512))
pad_area = X.shape[0]
X_batch[0][maxlength - pad_area:][:] = X1
'''
Y_pre=model.predict_generator(generate1("/Users/adminadmin/Documents/mywork/master/AVEC2014_AudioVisual/test512.txt"),steps=50)
Y_test=generatelabel("/Users/adminadmin/Documents/mywork/master/AVEC2014_AudioVisual/test512.txt")

accuracy,accuracy1,confusion_matrix,confusion_matrix1=accuracy(Y_pre,Y_test)
print (accuracy)
print (confusion_matrix)
print("========")
print (accuracy1)
print (confusion_matrix1)
