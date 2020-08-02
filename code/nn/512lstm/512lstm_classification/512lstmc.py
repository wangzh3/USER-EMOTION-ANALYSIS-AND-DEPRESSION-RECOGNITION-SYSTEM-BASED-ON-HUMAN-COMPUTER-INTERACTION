from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
import random
import pandas as pd
root="/Users/adminadmin/Documents/mywork/master/"
savepath=root+"code/nn/512lstm_classification/lstmloss.txt"
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

def changelable(dep):
    if (dep>=0 and dep < 14):
        dep1 = 0
    else:
        dep1 = 1
    return dep1

def generate1(traintxt):
    patharray=np.loadtxt(traintxt,dtype=np.str)
    patharray1=patharray.tolist()
    random.shuffle(patharray1)
    while True:
        for pathname in patharray1:
            filename=pathname.split("/")[-1]
            tail = filename.split(".")[1]
            head = filename.split(".")[0]
            if (tail == "txt"):
                X=np.loadtxt(pathname)
                X1=X.reshape(1, X.shape[0], 512)
                X_batch = np.zeros((1, maxlength, 512))
                pad_area = X.shape[0]
                X_batch[0][maxlength - pad_area:][:] = X1
                depvalue = changelable(int(head.split("_")[2]))
                depvalue_array = np.zeros(1, dtype=np.int)
                depvalue_array[0] = depvalue
                depvalue_array1=keras.utils.to_categorical(depvalue_array,num_classes=2)
                yield (X_batch, depvalue_array1)

BS=30
EP=2
model = Sequential()
model.add(LSTM(70,input_shape=(maxlength, 512),name="1"))
model.add(Dense(2,name="2"))
model.add(Activation('softmax',name="3"))
#sgd = keras.optimizers.SGD(lr=1e-6, decay=1e-4, momentum=0.9, nesterov=True)
sgd=keras.optimizers.Adamax(learning_rate=0.001)
checkpoint=ModelCheckpoint(root+"code/nn/512lstm_classification/lstmsave512/weights_{epoch:02d}.hdf5")

model.compile(loss='binary_crossentropy',  #class_weight #binary cross entropy losss
              optimizer=sgd,
              metrics=['accuracy'])
'''
dense1_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer('3').output)

dense1_output = dense1_layer_model.predict_generator(generate1("/Users/adminadmin/Documents/mywork/master/AVEC2014_AudioVisual/test512.txt"),steps=50)
print (dense1_output)
print (type(dense1_output))
'''

history = model.fit_generator(generate1(root+"AVEC2014_AudioVisual/train512.txt"),steps_per_epoch=int(total_train/BS),epochs=EP,
                              validation_data=generate1(root+"AVEC2014_AudioVisual/val512.txt"), validation_steps=int(Ntest/2),callbacks=[checkpoint])

#val_loss val_accuracy loss accuracy

result=pd.DataFrame(history.history).values
np.savetxt(savepath,result)
