import cv2
import dlib
import numpy as np
from module import visualize
netpath="/Users/adminadmin/documents/mywork/master/code/module/FER2013_VGG19/PrivateTest_model.t7"
cap = cv2.VideoCapture(0)
while(True):
    ret, img = cap.read()
    #cv2.imshow('img',img)

    detector = dlib.get_frontal_face_detector()
    face = detector(img, 1)
    if (len(face)!=0):
        for k, d in enumerate(face):
            height = d.bottom() - d.top()
            width = d.right() - d.left()
            face_matrix = np.zeros([height, width, 3], np.uint8)
            for m in range(height):
                for n in range(width):
                    face_matrix[m][n][0] = img[d.top() + m][d.left() + n][0]
                    face_matrix[m][n][1] = img[d.top() + m][d.left() + n][1]
                    face_matrix[m][n][2] = img[d.top() + m][d.left() + n][2]
            print(face_matrix.shape)
            r,r1= visualize.net7(face_matrix, netpath)
            print(r)
            print (r1)
            vec=visualize.net(face_matrix, netpath)
            print (vec)
    else:
        print("no face here")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
