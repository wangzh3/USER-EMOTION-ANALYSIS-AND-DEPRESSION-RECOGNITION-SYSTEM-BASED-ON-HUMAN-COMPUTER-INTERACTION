import cv2
import dlib
import numpy as np
import os
from module import visualize
import time
root="/Users/adminadmin/documents/mywork/master/"
netpath = root+"code/module/FER2013_VGG19/PrivateTest_model.t7"
gbottom=0
gtop=0
gright=0
gleft=0
def face2vec(imgfile):
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
                    # cv2.imwrite("test.jpg",face_matrix)
            arr = visualize.net(face_matrix, netpath)
            gbottom = d.bottom()
            gtop = d.top()
            gright = d.right()
            gleft = d.left()
            return arr.detach().numpy()
        else:
            return face2vec1(imgfile)

def face2vec1(imgfile):
    img = cv2.imread(imgfile)
    height = gbottom - gtop
    width = gright - gleft
    face_matrix = np.zeros([height, width, 3], np.uint8)
    for m in range(height):
        for n in range(width):
            face_matrix[m][n][0] = img[gtop + m][gleft + n][0]
            face_matrix[m][n][1] = img[gtop + m][gleft + n][1]
            face_matrix[m][n][2] = img[gtop + m][gleft + n][2]
    #cv2.imwrite("test1.jpg", face_matrix)
    arr = visualize.net(face_matrix, netpath)
    return arr.detach().numpy()

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
    for i in range(1, howmany(imagepath)+1):
        image=imagepath+str(i)+".jpg"
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

def save2vec(nparray,vecsavefile,judge):
    if (os.path.exists(judge)):
        f=open(vecsavefile,"w")
        np.savetxt(vecsavefile,nparray)
        f.close()
    else:
        os.mkdir(judge)
        f = open(vecsavefile, "w")
        np.savetxt(vecsavefile, nparray)
        f.close()

def totalimage(img_folder):
    n=0
    for image_folder in os.listdir(img_folder):
        name=image_folder.split("_")[-1]
        if (name== "video"):
            n+=1
    return n

def saveloop(img_folder,vectorpath,type1,total):
    n1 = 0
    for image_folder in os.listdir(img_folder):
        name = image_folder.split("_")[-1]
        if (name== "video"):
            thisvideoimage = img_folder + image_folder + "/"
            n1+=1

            end = endframe(thisvideoimage)
            start = startframe(thisvideoimage)
            nvector = looptime(start, end)
            print("==================" + str(nvector)+ type1 + "==================")
            n = 1
            i1 = start
            while (n <= nvector):
                print (str(n)+"/" +str(nvector)+" "+ str(n1)+"/"+str(total))
                img1 = thisvideoimage + str(i1) + ".jpg"
                print (img1)

                # get vector

                if (existface(img1) == True):
                    vec1 = face2vec(img1)
                else:
                    vec1 = face2vec1(img1)

                savefile = vectorpath + type1+"/" + image_folder + "/" + str(i1) + ".txt"
                judge = vectorpath + type1+"/" + image_folder + "/"
                save2vec(vec1, savefile, judge)
                print(savefile)

                i1 += 1
                n += 1

def save(imagepath,vectorpath):
    Freeform_img_folder=imagepath+"Freeform/"
    Northwind_img_folder=imagepath+"Northwind/"
    total1 = totalimage(Freeform_img_folder)
    total2 = totalimage(Northwind_img_folder)
    saveloop(Freeform_img_folder,vectorpath, "Freeform",total1 )
    saveloop(Northwind_img_folder,vectorpath, "Northwind",total2)

print ("start")

imagepath=root+"AVEC2014_AudioVisual/Image/Training/"
vectorpath=root+"AVEC2014_AudioVisual/Image2vec/Training/"
save(imagepath,vectorpath)

imagepath=root+"AVEC2014_AudioVisual/Image/Development/"
vectorpath=root+"AVEC2014_AudioVisual/Image2vec/Development/"
save(imagepath,vectorpath)

imagepath=root+"AVEC2014_AudioVisual/Image/Testing/"
vectorpath=root+"AVEC2014_AudioVisual/Image2vec/Testing/"
save(imagepath,vectorpath)

