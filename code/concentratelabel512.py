import numpy as np
import os
import pandas

root = "/Users/adminadmin/documents/mywork/master/"

def totalimage(img_folder):
    n=0
    for image_folder in os.listdir(img_folder):
        name=image_folder.split("_")[-1]
        if (name== "video"):
            n+=1
    return n

def minmax(imagepath):
    max=0
    min=float("INF")
    for frame in os.listdir(imagepath):
        name = frame.split(".")[1]
        if (name == "txt"):
            header = frame.split(".")[0]
            number = int(header)
            if (number > max):
                max = number
            if (number < min):
                min = number
    return int(min),int(max)

def loopsave(vectaskfoloderpath, Depfolderpath, savepath,total):
    number=1
    for vecfolder in os.listdir(vectaskfoloderpath):
        vecfolderpath=vectaskfoloderpath+vecfolder+"/"

        name=vecfolder.split("_")[-1]
        if (name == "video"):
            header1 = vecfolder.split("_")[0]
            header2 = vecfolder.split("_")[1]
            #find depression value
            for depfile in os.listdir(Depfolderpath):
                depname=depfile.split(".")[-1]
                if (depname == "csv"):
                    depfilearray = depfile.split("_")
                    if ((depfilearray[0] == header1) and (depfilearray[1] == header2)):
                        depfilename = Depfolderpath + depfile
                        depvalue = pandas.read_csv(depfilename, header=None).values[0][0]
                        break
            savename= savepath+header1+"_"+header2+"_"+ str(depvalue) +".txt"

            buffer = []
            min, max = minmax(vecfolderpath)
            # loop to save all vector
            for i in range(min,max+1): #vecfile in os.listdir(vecfolderpath):
                txtname=vecfolderpath+str(i)+".txt"
                vec = np.loadtxt(txtname).flatten()
                # loop in each element of the vector and save this vec to txt.
                buffer.append(vec)
            buffernp=np.array(buffer)
            np.savetxt(savename,buffernp)
            print (str(number)+"/"+str(total))
            print (savename)
            number+=1

def frame(labelpath,vecpath,labelsavepath):
    vectaskfoloderpath = vecpath + "Freeform/"
    Depfolder = labelpath + "Depression/"
    Fsavepath = labelsavepath + "Freeform/"
    total=totalimage(vectaskfoloderpath)
    loopsave(vectaskfoloderpath, Depfolder, Fsavepath,total)
    # "Northwind/"
    vectaskfoloderpath = vecpath + "Northwind/"
    Nsavepath = labelsavepath + "Northwind/"
    total = totalimage(vectaskfoloderpath)
    loopsave(vectaskfoloderpath, Depfolder, Nsavepath,total)

def frame_test(labelpath,vecpath,labelsavepath):
    vectaskfoloderpath = vecpath + "Freeform/"
    Depfolder = labelpath + "DepressionLabels/"
    Fsavepath = labelsavepath + "Freeform/"
    total = totalimage(vectaskfoloderpath)
    loopsave(vectaskfoloderpath, Depfolder, Fsavepath,total)
    # "Northwind/"
    vectaskfoloderpath = vecpath + "Northwind/"
    Nsavepath = labelsavepath + "Northwind/"
    total = totalimage(vectaskfoloderpath)
    loopsave(vectaskfoloderpath, Depfolder, Nsavepath,total)

labelpath = root + "AVEC2014_AudioVisual/Label/Training/"
vecpath=root+"AVEC2014_AudioVisual/Image2vec/Training/"
labelpath1 = root + "AVEC2014_AudioVisual/Label512/Training/"
frame(labelpath,vecpath, labelpath1)

labelpath = root + "AVEC2014_AudioVisual/Label/Development/"
vecpath=root+"AVEC2014_AudioVisual/Image2vec/Development/"
labelpath1 = root + "AVEC2014_AudioVisual/Label512/Development/"
frame(labelpath,vecpath, labelpath1)

labelpath = root + "AVEC2014_AudioVisual/Label/Testing/"
vecpath=root+"AVEC2014_AudioVisual/Image2vec/Testing/"
labelpath1 = root + "AVEC2014_AudioVisual/Label512/Testing/"
frame_test(labelpath,vecpath, labelpath1)