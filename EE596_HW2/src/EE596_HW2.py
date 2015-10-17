'''
Created on Oct 16, 2015

@author: Dan Sweet

EE596 Homework 2
'''
import cv2
import numpy as np
import os
from numpy import float32

max_iter = 10           #number of iterations to use in the K-means algorithm
epsilon = 1.0           #epsilon for accuracy of K-means algorithm
K = 5                   #Number of K-means clusters

IMG_TRAIN_PATH = '../face_training/'
IMG_TRAIN_GT_PATH = '../face_training_groundtruth/'

img_train = []
img_train_gt = []
ivec_train = []
fvec_train = []
center_train = []
class_train = []

kmean_img = []

#Read all training images into list
for filename in os.listdir(IMG_TRAIN_PATH):
    img = cv2.imread(os.path.join(IMG_TRAIN_PATH, filename))
    if img is not None:
        img_train.append(img)

#Read all training groundtruth images into list
for filename in os.listdir(IMG_TRAIN_GT_PATH):
    img = cv2.imread(os.path.join(IMG_TRAIN_GT_PATH, filename))
    if img is not None:
        img_thresh =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_train_gt.append(img_thresh)

#Reshape all images  
for x in img_train:
    ivec_train.append(x.reshape((-1,3)))

#Convert to float for K-means
for y in ivec_train:
    fvec_train.append(np.float32(y))

#Create K-Means Criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)

#Perform K-Means on all training images
pixel_valkmean_img = []
i = 0
for fvec in fvec_train:
    ret, labels, centers = cv2.kmeans(fvec, K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)     #Get K-Means
    #res = labels.reshape((img_train[i].shape))
    cent = np.uint8(centers)                                                                #Convert centers from float to uint8
    #center_train.append(cent)                                                               #Save Centers for later during training
    res = cent[labels.flatten()]                                                            #Create new image based on center labels
    res2 = res.reshape((img_train[i].shape))                                                #Reshape to original image shape
    
    xyz = labels.flatten()
    xyz2 = xyz.reshape((img_train[i].shape[0],img_train[i].shape[1]))

    for c in centers:
        center_train.append(c)
#     cv2.imshow('Image: '+str(i),res2)
#     cv2.waitKey()
    i += 1
    
    kmean_img.append(xyz2)
    
# cv2.destroyAllWindows()

# 10 if x == 0 else 0 for x in kmean_img[0]
# Set up for Classifying
# for k in range(0,K):
#     abc = kmean_img[0] & img_train_gt[0]
#     cv2.imshow('overlap',abc)
#     cv2.waitKey()

bayes_model = cv2.NormalBayesClassifier()

for i in range(len(kmean_img)):
    # Count how many of each cluster are in image
    region_count = [0]*K
    overlap_count = [0]*K
    class_label = [0]*K
    for x in range(kmean_img[i].shape[0]):
        for y in range(kmean_img[i].shape[1]):
            region_count[kmean_img[i][x,y]] += 1
            if (kmean_img[i][x,y] & img_train_gt[i][x,y]):
                overlap_count[kmean_img[i][x,y]] += 1
    #Check Strength of match:
    a = np.array(overlap_count, dtype = np.float)
    b = np.array(region_count, dtype = np.float)
    ratio = a/b
    
    x = 0
    for element in ratio:
        if (element >= 0.50):
            class_label[x] = 1
            class_train.append(1)
        else:
            class_label[x] = 0
            class_train.append(0)
        x+=1
    
    #class_train.append(class_label)    
    
#     print "Image Number: " + str(i+1)
#     print overlap_count
#     print region_count
#     print ratio
#     print class_label


# fjdaio = np.asarray(center_train)
# print center_train
# a1 = np.asarray(center_train)
# print a1
# a2 = np.float32(center_train)
# print a2
# b1 = np.asarray(class_train)
# b2 = b1.astype(int)
# 
# flattened = [val for sublist in a2 for val in sublist]
# # print flattened
# qwerty = np.array(a2)
# asdf = np.array([0.0,0.0,0.0])
# b2 = np.array(class_train)
#print qwerty

bayes_model.train(np.array(np.float32(center_train)),cv2.CV_ROW_SAMPLE,np.array(class_train))




# for element in ratio:
#     if()

# overlap_count = []
# 
# for pixel in kmean_img[0]:
# #    if kmean_img[0][pixel] & img_train_gt[0][pixel]:
#         area_count[kmean_img[0][pixel]] = area_count[kmean_img[0][pixel]] + 1
#         

# cv2.imshow('ackvjla', img_train_gt[0])





# cv2.imshow('overlap',kmean_img[0])
# cv2.imshow('overlap',abc)
# cv2.waitKey()
cv2.destroyAllWindows()

#ret, labels, centers = cv2.kmeans(fvec_train[0], K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

#labels.reshape(100,100)
# xyz = labels.reshape(img_train[1].shape[0],img_train[1].shape[1])
# np.uint8(xyz)

# cent = np.uint8(centers)
# res = cent[labels.flatten()]
# res2 = res.reshape((img_train[0].shape))
# cv2.imshow('res2',res2)
# cv2.waitKey()
# cv2.destroyAllWindows()

x = 0



