import os
import numpy as np
root="/Users/adminadmin/Documents/mywork/master/"
'''
def totxt1(path,path1,path2,path3): #generate txt path A D V for pytorch to predict adv
    file = open(path2, "w")
    file1 = open(path3, "w")
    i=0
    for videofolder in os.listdir(path):
        if (videofolder  != ".DS_Store"):
           txtpath=path+videofolder+"/"
           for txtfile in os.listdir(txtpath):
               if (txtfile != ".DS_Store"):
                   txtfilepath=txtpath+txtfile
                   print (txtfilepath)

                   videofolderarray=videofolder.split("_")
                   name0=videofolderarray[0]
                   name1 = videofolderarray[1]

                   for labeltxt in os.listdir(path1):
                       if (labeltxt != ".DS_Store"):

                           labeltxtarray1=labeltxt.split(".")
                           labeltxtarray2 = labeltxtarray1[0].split("_")
                           if (name0==labeltxtarray2[0] and name1==labeltxtarray2[1]):

                               labeltxtpath=path1+labeltxt
                               print (labeltxtpath)
                               m=np.loadtxt(labeltxtpath,dtype=np.str)
                               number=int(txtfile.split(".")[0])
                               index=number-1
                               A=m[index][0]
                               D=m[index][1]
                               V=m[index][2]

                               if(i>75000):
                                   file1.write(txtfilepath + " " + A + " " + D + " " + V + "\n")
                               else:
                                   file.write(txtfilepath + " " + A + " " + D + " " + V + "\n")
                               i += 1

    file.close()


path=root+"AVEC2014_AudioVisual/Vector/Training/Freeform/"
path1=root+"AVEC2014_AudioVisual/Label2/Training/Freeform/"
path2=root+"AVEC2014_AudioVisual/Train.txt"
path3=root+"AVEC2014_AudioVisual/Test.txt"
#totxt(path,path1,path2)
totxt1(path,path1,path2,path3)
'''
#generate txt path for train dev test for keras to predict by 512
trainF=root+"AVEC2014_AudioVisual/Label512/Training/Freeform/"
trainN=root+"AVEC2014_AudioVisual/Label512/Training/Northwind/"
devF=root+"AVEC2014_AudioVisual/Label512/Development/Freeform/"
devN=root+"AVEC2014_AudioVisual/Label512/Development/Northwind/"
testF=root+"AVEC2014_AudioVisual/Label512/Testing/Freeform/"
testN=root+"AVEC2014_AudioVisual/Label512/Testing/Northwind/"

def generate_train(traintxt):
    trainfile=open(traintxt,"w")
    for file in os.listdir(trainF):
        item = file.split(".")
        if (item[1] == "txt"):
            txtpath = trainF + file
            print (txtpath)
            trainfile.write(txtpath+"\n")

    for file in os.listdir(trainN):
        item = file.split(".")
        if (item[1] == "txt"):
            txtpath = trainN + file
            print(txtpath)
            trainfile.write(txtpath+"\n")

    for file in os.listdir(devF):
        item = file.split(".")
        if (item[1] == "txt"):
            txtpath = devF + file
            print(txtpath)
            trainfile.write(txtpath+"\n")

    for file in os.listdir(devN):
        item = file.split(".")
        if (item[1] == "txt"):
            txtpath = devN + file
            print(txtpath)
            trainfile.write(txtpath+"\n")
    trainfile.close()

def generate_val(valtxt,testtxt):
    file_val=open(valtxt,"w")
    file_test=open(testtxt,"w")
    for file in os.listdir(testF):
        item = file.split(".")
        if (item[1] == "txt"):
            header=int(file.split("_")[0])
            txtpath = testF + file

            if (header<250):
                print(txtpath)
                file_val.write(txtpath+"\n")
            else:
                print(txtpath)
                file_test.write(txtpath+"\n")

    for file in os.listdir(testN):
        item = file.split(".")
        if (item[1] == "txt"):
            header = int(file.split("_")[0])
            txtpath = testN + file

            if (header<250):
                print(txtpath)
                file_val.write(txtpath+"\n")
            else:
                print(txtpath)
                file_test.write(txtpath+"\n")
    file_val.close()
    file_test.close()

traintxt=root+"AVEC2014_AudioVisual/train512.txt"
valtxt=root+"AVEC2014_AudioVisual/val512.txt"
testtxt=root+"AVEC2014_AudioVisual/test512.txt"
generate_train(traintxt)
generate_val(valtxt,testtxt)










