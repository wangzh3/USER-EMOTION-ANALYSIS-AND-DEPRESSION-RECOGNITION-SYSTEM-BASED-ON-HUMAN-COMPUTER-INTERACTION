import cv2
import dlib
import numpy as np
import os
from module import visualize
import csv
import pandas as pd
root="/Users/adminadmin/documents/mywork/master/"
netpath = root+"code/module/FER2013_VGG19/PrivateTest_model.t7"
gbottom=0
gtop=0
gright=0
gleft=0
def face2emo(imgfile):
    global gbottom,gtop,gright,gleft
    img = cv2.imread(imgfile)
    detector = dlib.get_frontal_face_detector()
    face = detector(img, 1)
    for k, d in enumerate(face):
        height = d.bottom() - d.top()
        width = d.right() - d.left()
        face_matrix = np.zeros([height, width, 3], np.uint8)
        if (d.top() + height < img.shape[0] and d.left() + width < img.shape[1]):
            for m in range(height):
                for n in range(width):
                    face_matrix[m][n][0] = img[d.top() + m][d.left() + n][0]
                    face_matrix[m][n][1] = img[d.top() + m][d.left() + n][1]
                    face_matrix[m][n][2] = img[d.top() + m][d.left() + n][2]
            emocode,emo = visualize.net7(face_matrix, netpath)
            gbottom = d.bottom()
            gtop = d.top()
            gright = d.right()
            gleft = d.left()
            return emocode,emo
        else:
            return face2emo1(imgfile)

def face2emo1(imgfile):
    img = cv2.imread(imgfile)
    height = gbottom - gtop
    width = gright - gleft
    face_matrix = np.zeros([height, width, 3], np.uint8)
    for m in range(height):
        for n in range(width):
            face_matrix[m][n][0] = img[gtop + m][gleft + n][0]
            face_matrix[m][n][1] = img[gtop + m][gleft + n][1]
            face_matrix[m][n][2] = img[gtop + m][gleft + n][2]
    emocode, emo = visualize.net7(face_matrix, netpath)
    return emocode, emo

#imagepath="/Users/adminadmin/Documents/mywork/master/AVEC2014_AudioVisual/Image/Training/Freeform/203_1_Freeform_video/"
#find how many frame do we have for 1 video
def howmany(imagepath):
    n=0
    for frame in os.listdir(imagepath):
        name=frame.split(".")[1]
        if (name== "jpg"):
            n+=1
    return n

#loop time for each video
#How many 512x5 can we get
def looptime(start,end):
    length=end-start+1
    return length

def existface(imgfile):
    img = cv2.imread(imgfile)
    detector = dlib.get_frontal_face_detector()
    face = detector(img, 1)
    if (len(face) != 0):  # have face
        for k, d in enumerate(face):
            height = d.bottom() - d.top()
            width = d.right() - d.left()
            if (d.top() + height < img.shape[0] and d.left() + width < img.shape[1]): #over bounded
                return True
            else:
                return False
    else:# no face
        return False

def startframe(imagepath):
    for i in range(100, howmany(imagepath)+1):
        image=imagepath+str(i)+".jpg"
        print (i)
        if (existface(image) == True):  # have face
            return i

def endframe(imagepath):
    max=0
    for frame in os.listdir(imagepath):
        name = frame.split(".")[1]
        if (name == "jpg"):
            header = frame.split(".")[0]
            number = int(header)
            if (number > max):
                max = number
    return max

def totalimage(img_folder):
    n=0
    for image_folder in os.listdir(img_folder):
        name=image_folder.split("_")[-1]
        if (name== "video"):
            n+=1
    return n

def video_emo_series(frame_folder,nth_video,total_video):
    end = endframe(frame_folder)
    start = startframe(frame_folder)
    nvector = looptime(start, end)
    emo_number=np.zeros(7,np.float)

    n = 1 #nth image
    i1 = start #start point
    while (n <= nvector):  # change to nvector
        print(str(n) + "/" + str(nvector)+" "+str(nth_video)+"/"+str(total_video))
        img1 = frame_folder + str(i1) + ".jpg"
        print(img1)
        # get emotion
        if (existface(img1) == True):
            #print("yes")
            emo_code,emotion = face2emo(img1)
        else:
            #print ("no")
            emo_code, emotion = face2emo1(img1)

        emo_number[emo_code]+=1
        i1 += 1
        n += 1
    normalize=emo_number / nvector
    normalize1=normalize.astype(np.str)
    return normalize1

def find_depression(header1,header2, Depfolderpath):
    # find depression value
    for depfile in os.listdir(Depfolderpath):
        depname = depfile.split(".")[-1]
        if (depname == "csv"):
            depfilearray = depfile.split("_")
            if ((depfilearray[0] == header1) and (depfilearray[1] == header2)):
                depfilename = Depfolderpath + depfile
                depvalue = pd.read_csv(depfilename, header=None).values[0][0]
                return depvalue

def saveloop(csvpath):
    class_names = ['Tester','Experiment','Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral',"Depression"]
    #class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    img_folder_task=[root+"AVEC2014_AudioVisual/Image/Training/Freeform/",
                     root + "AVEC2014_AudioVisual/Image/Training/Northwind/",
                     root + "AVEC2014_AudioVisual/Image/Development/Freeform/",
                     root + "AVEC2014_AudioVisual/Image/Development/Northwind/"
                     ]
    depfolder_task=[root + "AVEC2014_AudioVisual/Label/Training/Depression/",
                    root + "AVEC2014_AudioVisual/Label/Training/Depression/",
                    root + "AVEC2014_AudioVisual/Label/Development/Depression/",
                    root + "AVEC2014_AudioVisual/Label/Development/Depression/"
                    ]
    with open(csvpath, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(class_names)
        for i in range(4):
            img_folder=img_folder_task[i]
            depfolder=depfolder_task[i]
            total = totalimage(img_folder)
            n1 = 1  # nth video
            for image_folder in os.listdir(img_folder):
                name = image_folder.split("_")[-1]
                if (name == "video"):
                    queue = []
                    tester = image_folder.split("_")[0] + "_" + image_folder.split("_")[1]
                    experiment = image_folder.split("_")[2]
                    queue.append(tester)
                    queue.append(experiment)
                    thisvideoimage = img_folder + image_folder + "/"
                    vec = video_emo_series(thisvideoimage, n1, total)
                    for item in vec:
                        queue.append(item)
                    header1 = image_folder.split("_")[0]
                    header2 = image_folder.split("_")[1]
                    # find depression
                    depression = str(find_depression(header1, header2, depfolder))
                    queue.append(depression)
                    writer.writerow(queue)
                    n1 += 1


#csvpath = root + "code/relation.csv"
#saveloop(csvpath)

def savelooptest(csvpath):
    class_names = ['Tester','Experiment','Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral',"Depression"]
    #class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    img_folder_task=[root+"AVEC2014_AudioVisual/Image/Testing/Freeform/",
                     root + "AVEC2014_AudioVisual/Image/Testing/Northwind/"
                     ]
    depfolder=root + "AVEC2014_AudioVisual/Label/Testing/DepressionLabels/"

    with open(csvpath, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(class_names)
        for i in range(4):
            img_folder=img_folder_task[i]
            total = totalimage(img_folder)
            n1 = 1  # nth video
            for image_folder in os.listdir(img_folder):
                name = image_folder.split("_")[-1]
                if (name == "video"):
                    queue = []
                    tester = image_folder.split("_")[0] + "_" + image_folder.split("_")[1]
                    experiment = image_folder.split("_")[2]
                    queue.append(tester)
                    queue.append(experiment)
                    thisvideoimage = img_folder + image_folder + "/"
                    vec = video_emo_series(thisvideoimage, n1, total)
                    for item in vec:
                        queue.append(item)
                    header1 = image_folder.split("_")[0]
                    header2 = image_folder.split("_")[1]
                    # find depression
                    depression = str(find_depression(header1, header2, depfolder))
                    queue.append(depression)
                    writer.writerow(queue)
                    n1 += 1

csvpath = root + "code/test.csv"
savelooptest(csvpath)