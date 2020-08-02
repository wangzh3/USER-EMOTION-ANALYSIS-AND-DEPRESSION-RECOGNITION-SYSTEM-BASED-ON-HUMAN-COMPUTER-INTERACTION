import cv2
import numpy as np
import os
import pandas

root = "/Users/adminadmin/documents/mywork/master/"


def combine(A1, D1):
    if (A1.shape[0] == D1.shape[0]):
        L1 = np.hstack((A1, D1))
    elif ((A1.shape[0] > D1.shape[0])):
        line = D1.shape[0]
        A2 = A1[0:line, :]

        L1 = np.hstack((A2, D1))
    else:
        line = A1.shape[0]
        D2 = D1[0:line, :]
        L1 = np.hstack((A1, D2))
    return L1

def save(nparray, savepath, name):
    save = savepath + name + ".txt"
    f = open(save, "w")
    np.savetxt(save, nparray)
    f.close()
    print(save)

def loopsave(labelA, labelD, labelV, Dep, savepath1):
    for csvfileA in os.listdir(labelA):
        print("=================================" + csvfileA + "=================================")
        if (csvfileA != ".DS_Store"):
            csvfilearray = csvfileA.split("_")
            csvA0 = csvfilearray[0]
            csvA1 = csvfilearray[1]
            A = pandas.read_csv(labelA + csvfileA, header=None).values
            A1 = A[:, 0].reshape((-1, 1))

            for csvfileD in os.listdir(labelD):
                if (csvfileD != ".DS_Store"):
                    csvfilearray = csvfileD.split("_")
                    csvD0 = csvfilearray[0]
                    csvD1 = csvfilearray[1]
                    if ((csvD0 == csvA0) and (csvD1 == csvA1)):
                        D = pandas.read_csv(labelD + csvfileD, header=None).values
                        D1 = D[:, 0].reshape((-1, 1))

                        L1 = combine(A1, D1)

                        for csvfileV in os.listdir(labelV):
                            if (csvfileV != ".DS_Store"):
                                csvfilearray = csvfileV.split("_")
                                csvV0 = csvfilearray[0]
                                csvV1 = csvfilearray[1]
                                if ((csvV0 == csvA0) and (csvV1 == csvA1)):
                                    V = pandas.read_csv(labelV + csvfileV, header=None).values
                                    V1 = V[:, 0].reshape((-1, 1))

                                    L2 = combine(L1, V1)
                                    # print(L2)

                                    for depfile in os.listdir(Dep):
                                        if (depfile != ".DS_Store"):
                                            depfilearray = depfile.split("_")
                                            if ((depfilearray[0] == csvA0) and (depfilearray[1] == csvA1)):
                                                depfilename = Dep + depfile
                                                depvalue = pandas.read_csv(depfilename, header=None).values[0][0]
                                                name = depfile.split(".")[0]
                                                namearray = name.split("_")
                                                newname = namearray[0] + "_" + namearray[1] + "_" + str(depvalue)
                                                save(L2, savepath1, newname)

def loopsavetest(labelfoloder, Depfolder, savepath1,type1):
    for csvfileA in os.listdir(labelfoloder):
        print("=================================" + csvfileA + "=================================")
        if (csvfileA != ".DS_Store"):
            csvfileA1=csvfileA.replace("-","_")
            csvfilearray = csvfileA1.split("_")
            csvA0 = csvfilearray[0]
            csvA1 = csvfilearray[1]
            type2=csvfilearray[2]
            advtype=csvfilearray[3].split(".")[0]
            if(type2==type1):
                print (advtype)
                if(advtype=="AROUSAL"):
                    A = pandas.read_csv(labelfoloder + csvfileA, header=None).values
                    A1 = A[:, 0].reshape((-1, 1))

                    for csvfileD in os.listdir(labelfoloder):
                        if (csvfileD != ".DS_Store"):
                            csvfileD1 = csvfileD.replace("-", "_")
                            csvfilearray = csvfileD1.split("_")
                            csvD0 = csvfilearray[0]
                            csvD1 = csvfilearray[1]
                            type2 = csvfilearray[2]
                            advtype = csvfilearray[3].split(".")[0]
                            if ((type2==type1) and (advtype=="DOMINANCE") and (csvD0 == csvA0) and (csvD1 == csvA1)):
                                D = pandas.read_csv(labelfoloder + csvfileD, header=None).values
                                D1 = D[:, 0].reshape((-1, 1))
                                L1 = combine(A1, D1)

                                for csvfileV in os.listdir(labelfoloder):
                                    if (csvfileV != ".DS_Store"):
                                        csvfileV1 = csvfileV.replace("-", "_")
                                        csvfilearray = csvfileV1.split("_")
                                        csvV0 = csvfilearray[0]
                                        csvV1 = csvfilearray[1]
                                        type2 = csvfilearray[2]
                                        advtype = csvfilearray[3].split(".")[0]
                                        if ((type2==type1) and (advtype=="VALENCE") and(csvV0 == csvA0) and (csvV1 == csvA1)):
                                            V = pandas.read_csv(labelfoloder + csvfileV, header=None).values
                                            V1 = V[:, 0].reshape((-1, 1))

                                            L2 = combine(L1, V1)

                                            for depfile in os.listdir(Depfolder):
                                                if (depfile != ".DS_Store"):
                                                    depfilearray = depfile.split("_")
                                                    if ((depfilearray[0] == csvA0) and (depfilearray[1] == csvA1)):
                                                        depfilename = Depfolder + depfile
                                                        depvalue = pandas.read_csv(depfilename, header=None).values[0][0]

                                                        name = depfile.split(".")[0]
                                                        namearray = name.split("_")
                                                        newname = namearray[0] + "_" + namearray[1] + "_" + str(depvalue)
                                                        save(L2, savepath1, newname)

def frame(labelpath,labelpath1,mark):
    if (mark == "train"):
        Freeform_labelA = labelpath + "Freeform/Arousal/"
        Freeform_labelD = labelpath + "Freeform/Dominance/"
        Freeform_labelV = labelpath + "Freeform/Valence/"
        Freeform_Dep = labelpath + "Depression/"
        Fsavepath = labelpath1 + "Freeform/"

        loopsave(Freeform_labelA, Freeform_labelD, Freeform_labelV, Freeform_Dep, Fsavepath)

        Northwind_labelA = labelpath + "Northwind/Arousal/"
        Northwind_labelD = labelpath + "Northwind/Dominance/"
        Northwind_labelV = labelpath + "Northwind/Valence/"
        Northwind_Dep = labelpath + "Depression/"
        Nsavepath = labelpath1 + "Northwind/"
        loopsave(Northwind_labelA, Northwind_labelD, Northwind_labelV, Northwind_Dep, Nsavepath)

    if (mark == "test"):
        affectfolder=labelpath+"AffectLabels/"
        depfolder=labelpath+"DepressionLabels/"
        Fsavepath = labelpath1 + "Freeform/"
        loopsavetest(affectfolder, depfolder, Fsavepath, "Freeform")

        Nsavepath = labelpath1 + "Northwind/"
        loopsavetest(affectfolder, depfolder, Nsavepath, "Northwind")


labelpath = root + "AVEC2014_AudioVisual/Label/Training/"
labelpath1 = root + "AVEC2014_AudioVisual/Label2/Training/"
frame(labelpath, labelpath1,"train")

labelpath = root + "AVEC2014_AudioVisual/Label/Development/"
labelpath1 = root + "AVEC2014_AudioVisual/Label2/Development/"
frame(labelpath,labelpath1, "train")

labelpath = root + "AVEC2014_AudioVisual/Label/Testing/"
labelpath1 = root + "AVEC2014_AudioVisual/Label2/Testing/"
frame(labelpath, labelpath1,"test")