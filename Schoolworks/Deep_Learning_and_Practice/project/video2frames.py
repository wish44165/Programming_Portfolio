# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 18:40:57 2021

@author: ktpss

department: IMM
student number: 309653012
name: yuhsi, Chen
"""

import cv2
import numpy as np


"""
totalFrames
dataset1
cam0: 20 / 5334
cam1: 20 / 4941
cam2: 20 / 8016
cam3: 20 / 4048
dataset2
cam0: 20 / 
.
.
.


dataset5
cam0: 40 / 20969

cam1: test

cam4: 40 / 21522
"""

cam_n = 4
dataset_n = 5

"""
with open('C:/Users/ktpss/Desktop/Deep_Learning_and_Pratice/project presentation/drone-tracking-datasets/dataset'+str(int(dataset_n))+'/detections/cam'+str(int(cam_n))+'_.txt') as f:
    lines = f.readlines()

nonEmpty = []
for i in range(len(lines)):
    l = [float(x) for x in lines[i].split()]
    if l[1] + l[2] != 0:
        nonEmpty.append(i)



# Opens the Video file

N = len(nonEmpty) // 20    # take 1 frame per N frames

cap= cv2.VideoCapture('C:/Users/ktpss/Desktop/Deep_Learning_and_Pratice/project presentation/drone-tracking-datasets/dataset'+str(int(dataset_n))+'/cam'+str(int(cam_n))+'/cam'+str(int(cam_n))+'.mp4')
i = 0
ct = 0    # number of figure
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i in nonEmpty:
        ct+=1
        if ct%N==0:
            cv2.imwrite('C:/Users/ktpss/Desktop/Deep_Learning_and_Pratice/project presentation/drone-tracking-datasets/dataset'+str(int(dataset_n))+'/cam'+str(int(cam_n))+'/labelimg_cam'+str(int(cam_n))+'/d'+str(int(dataset_n))+'c'+str(int(cam_n))+'_'+str(int(ct/N))+'.jpg',frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()
"""


# Opens the Video file
cap= cv2.VideoCapture('C:/Users/ktpss/Desktop/Deep_Learning_and_Pratice/project presentation/drone-tracking-datasets/dataset'+str(int(dataset_n))+'/videos/cam'+str(int(cam_n))+'/cam'+str(int(cam_n))+'.mp4')
i=0
ct = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i%500==0:
        cv2.imwrite('C:/Users/ktpss/Desktop/Deep_Learning_and_Pratice/project presentation/drone-tracking-datasets/dataset'+str(int(dataset_n))+'/videos/cam'+str(int(cam_n))+'/labelimg_cam'+str(int(cam_n))+'/d'+str(int(dataset_n))+'c'+str(int(cam_n))+'_'+str(int(ct))+'.jpg',frame)
        ct+=1
    i+=1

cap.release()
cv2.destroyAllWindows()
