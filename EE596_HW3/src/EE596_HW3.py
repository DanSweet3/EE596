'''
from numpy import reshape
Created on Nov 3, 2015

@author: Dan Sweet

EE596 Homework #3
'''

import cv2
import numpy as np
import os
import sys

img_capri = cv2.imread('../Images/capri.jpg')

## SELECT ONE INPUT IMAGE PAIR HERE ##
# img1_in = cv2.imread('../Images/a2.jpg')
# img2_in = cv2.imread('../Images/a1.jpg')

img1_in = cv2.imread('../Images/b2.jpg')
img2_in = cv2.imread('../Images/b1.jpg')

# These are my own images:
# img1_in = cv2.imread('../Images/C2a.jpg')
# img2_in = cv2.imread('../Images/C1a.jpg')
# minHessian = 180

# Declare SURF and SIFT Objects
minHessian = 400
surf = cv2.SURF(minHessian)
sift = cv2.SIFT()

# - PART 1 on Capri Image - 
#HARRIS
gray_capri = cv2.cvtColor(img_capri,cv2.COLOR_BGR2GRAY)
gray_float = np.float32(gray_capri)
dst = cv2.cornerHarris(gray_float,2,3,0.04)
#Mark Corners
dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img_capri[dst>0.02*dst.max()]=[0,0,255]

kp_surf, des_surf = surf.detectAndCompute(gray_capri,None)
kp_sift, des_sift = sift.detectAndCompute(gray_capri,None)

# Display Part 1 Results
cv2.imshow("Harris Image",img_capri)
cv2.imshow("SURF Image",cv2.drawKeypoints(img_capri,kp_surf))
cv2.imshow("SIFT Image",cv2.drawKeypoints(img_capri,kp_sift))
cv2.waitKey()
cv2.destroyAllWindows()

# - PART 2 - Matching Descriptors and Compute Homography
img1 = cv2.cvtColor(img1_in,cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_in,cv2.COLOR_BGR2GRAY)

#IF SURF:
label = "SURF "
kp1, des1 = surf.detectAndCompute(img1,None)
kp2, des2 = surf.detectAndCompute(img2,None)

#IF SIFT:
# label = "SIFT "
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)

#DRAW KEYPOINTS:
cv2.imshow(label + "Right Image",cv2.drawKeypoints(img1_in,kp1))
cv2.imshow(label + "Left Image",cv2.drawKeypoints(img2_in,kp2))
cv2.waitKey()
#cv2.destroyAllWindows()

#PRINT A FEW KEYPOINTS:
print "KeyPoints:"
print "KP1(0): Loc: "+ str(kp1[0].pt) +", Angle: "+str(kp1[0].angle) +", Response: "+str(kp1[0].response)+", Size: "+str(kp1[0].size)
print "KP1(100): Loc: "+ str(kp1[100].pt) +", Angle: "+str(kp1[100].angle) +", Response: "+str(kp1[100].response)+", Size: "+str(kp1[100].size)
print "KP2(0): Loc: "+ str(kp2[0].pt) +", Angle: "+str(kp2[0].angle) +", Response: "+str(kp2[0].response)+", Size: "+str(kp2[0].size)
print "KP2(100): Loc: "+ str(kp2[100].pt) +", Angle: "+str(kp2[100].angle) +", Response: "+str(kp2[100].response)+", Size: "+str(kp2[100].size)

#PERFORM FLANN MATCHING
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.match(des1,des2)

#COLLECT MATCHED POINTS
src=[]
dst=[]
for m in matches:
    src.append(kp1[m.queryIdx].pt)
    dst.append(kp2[m.trainIdx].pt)

#PERFORM HOMOGRAPHY
M, mask = cv2.findHomography(np.array(src), np.array(dst), cv2.RANSAC)#,5.0)
print "Homography:"
print M

#Print first 5 mappings for report
for i in range(1,6):
    print "Element: "+str(i)
    print "Query Index: " + str(matches[i].queryIdx)
    print "Coordinates: " + str(src[i])

## PART 3 - WARPING AND COMBINING ##
warped = cv2.warpPerspective(img1_in, M, (img1_in.shape[1] + img2_in.shape[1], img1_in.shape[0]))
warped[0:img2_in.shape[0], 0:img2_in.shape[1], :] = img2_in
cv2.imshow("Panorama",warped)
cv2.waitKey()
cv2.destroyAllWindows()
