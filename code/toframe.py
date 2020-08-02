import cv2
import numpy as np
import os
import pandas

def frame_number(thisvideo,labelpath,mark):
    if(mark=="train"):
        Freeform_label = labelpath + "Freeform/Arousal/"
        Northwind_label = labelpath + "Northwind/Arousal/"
        thisvideoarray = thisvideo.split("_")
        name0 = thisvideoarray[0]
        name1 = thisvideoarray[1]
        type = thisvideoarray[2]
        if (type == "Freeform"):
            for csvfile in os.listdir(Freeform_label):
                if (csvfile != ".DS_Store"):
                    csvfilearray = csvfile.split("_")
                    csv0 = csvfilearray[0]
                    csv1 = csvfilearray[1]
                    if ((name0 == csv0) & (name1 == csv1)):
                        m = pandas.read_csv(Freeform_label + csvfile, header=None).values
                        return m.shape[0]

        if (type == "Northwind"):
            for csvfile in os.listdir(Northwind_label):
                if (csvfile != ".DS_Store"):
                    csvfilearray = csvfile.split("_")
                    csv0 = csvfilearray[0]
                    csv1 = csvfilearray[1]
                    if ((name0 == csv0) & (name1 == csv1)):
                        m = pandas.read_csv(Northwind_label + csvfile, header=None).values
                        return m.shape[0]

    if(mark=="test"):
        thisvideoarray = thisvideo.split("_")
        name0 = thisvideoarray[0]
        name1 = thisvideoarray[1]
        type = thisvideoarray[2]

        for csvfile in os.listdir(labelpath):
            if (csvfile != ".DS_Store"):
                csvfilereplace = csvfile.replace("-", "_")
                csvfilearray = csvfilereplace.split("_")
                csv0 = csvfilearray[0]
                csv1 = csvfilearray[1]
                csv2 = csvfilearray[2]
                if ((name0 == csv0) & (name1 == csv1) & (type == csv2)):
                    m = pandas.read_csv(labelpath + csvfile, header=None).values
                    return m.shape[0]


def frame_number_test(thisvideo,labelpath):
    thisvideoarray=thisvideo.split("_")
    name0=thisvideoarray[0]
    name1=thisvideoarray[1]
    type=thisvideoarray[2]

    for csvfile in os.listdir(labelpath):
        if (csvfile != ".DS_Store"):
            csvfilereplace = csvfile.replace("-","_")
            csvfilearray=csvfilereplace.split("_")
            csv0 = csvfilearray[0]
            csv1 = csvfilearray[1]
            csv2=csvfilearray[2]
            if ((name0 == csv0) & (name1 == csv1)&(type==csv2)):
                m = pandas.read_csv(labelpath + csvfile, header=None).values
                return m.shape[0]

def save2image(thisvideopath,imgsavepath,nframe,count,type):
    cap = cv2.VideoCapture(thisvideopath)
    i = 1
    while (cap.isOpened()):
        ret, frame = cap.read()
        if(os.path.exists(imgsavepath)):
            if(ret==True):
             cv2.imwrite(imgsavepath+str(i)+".jpg", frame)
        else:
            os.mkdir(imgsavepath)
            if (ret == True):
              cv2.imwrite(imgsavepath + str(i) + ".jpg", frame)
        print (imgsavepath+str(i)+".jpg")
        print(type+" "+str(count))
        i += 1
        if (i > nframe):
            break
    cap.release()

def save(videopath,labelpath,imagepath,mark):
    i=0
    j=0
    Freeform_video=videopath+"Freeform/"
    Northwind_video=videopath+"Northwind/"
    for video in os.listdir(Freeform_video):
        if (video != ".DS_Store"):
            i += 1
            print("=================="+str(i)+"Freeform ==================")
            thisvideopath=Freeform_video+video
            videoarray=video.split(".")
            videoname=videoarray[0]
            imgsavepath = imagepath + "Freeform/"+videoname + "/"
            nframe = frame_number(video, labelpath,mark)
            save2image(thisvideopath, imgsavepath, nframe,i,"Freeform")

    for video in os.listdir(Northwind_video):
        if (video != ".DS_Store"):
            j+=1
            print("==================" + str(j) + "Northwind ==================")
            thisvideopath=Northwind_video+video
            videoarray=video.split(".")
            videoname=videoarray[0]
            imgsavepath = imagepath + "Northwind/"+videoname + "/"
            nframe = frame_number(video, labelpath,mark)
            save2image(thisvideopath, imgsavepath, nframe,j,"Northwind")

videopath="/Users/adminadmin/Documents/mywork/master/AVEC2014_AudioVisual/Video/Training/"
labelpath="/Users/adminadmin/Documents/mywork/master/AVEC2014_AudioVisual/Label/Training/"
imagepath="/Users/adminadmin/Documents/mywork/master/AVEC2014_AudioVisual/Image/Training/"

save(videopath,labelpath,imagepath,"train")

videopath="/Users/adminadmin/Documents/mywork/master/AVEC2014_AudioVisual/Video/Development/"
labelpath="/Users/adminadmin/Documents/mywork/master/AVEC2014_AudioVisual/Label/Development/"
imagepath="/Users/adminadmin/Documents/mywork/master/AVEC2014_AudioVisual/Image/Development/"

save(videopath,labelpath,imagepath,"train")


videopath="/Users/adminadmin/Documents/mywork/master/AVEC2014_AudioVisual/Video/Testing/"
labelpath="/Users/adminadmin/Documents/mywork/master/AVEC2014_AudioVisual/Label/Testing/AffectLabels/"
imagepath="/Users/adminadmin/Documents/mywork/master/AVEC2014_AudioVisual/Image/Testing/"

save(videopath,labelpath,imagepath,"test")

