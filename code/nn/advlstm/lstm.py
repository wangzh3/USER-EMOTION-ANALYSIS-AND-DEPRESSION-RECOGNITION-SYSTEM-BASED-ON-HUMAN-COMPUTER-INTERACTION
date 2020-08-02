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
savepath=root+"code/nn/lstmloss.txt"
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
print (howmany())
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

BS=30
EP=2
model = Sequential()
model.add(LSTM(70,input_shape=(maxlength, 3),name="1"))
model.add(Dense(1,name="2"))
model.add(Activation('softplus',name="3"))
sgd = keras.optimizers.SGD(lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)
checkpoint=ModelCheckpoint(root+"code/nn/lstmsave/weights_{epoch:02d}.hdf5")

model.compile(loss='mean_squared_error',  #class_weight #binary cross entropy losss
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                        batch_size=BS,
                        epochs=EP,
                        validation_data=(X_val, Y_val), callbacks=[checkpoint])

file=open(savepath,"w")
file.write("epoch val_loss\n")
i=1
for item in history.history["val_loss"]:
    file.write(str(i)+" "+str(item))
    file.write("\n")
    i+=1
file.close()

'''
dense1_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer('3').output)

dense1_output = dense1_layer_model.predict(X_train)
print (dense1_output.shape)
print (type(dense1_output))
'''
