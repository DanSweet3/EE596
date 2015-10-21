'''
Created on Oct 16, 2015

@author: Dan Sweet

EE596 Homework 2
'''
import cv2
import numpy as np
import os
import sys
import time
from numpy import float32

start_time = time.time()

DO_RG = 1               #Set to 1 for RG space, 0 foSr RGB space
DO_BAYES = 0            # '0' = BAYES , '1' = RANDOM TREES
DEBUG_K_TRAIN = 0


# if DO_RG:
#     max_iter = 25           #number of iterations to use in the K-means algorithm
#     epsilon = 1           #epsilon for accuracy of K-means algorithm
#     K = 5                   #Number of K-means clusters
# else:
#     max_iter = 25           #number of iterations to use in the K-means algorithm
#     epsilon = 1           #epsilon for accuracy of K-means algorithm
#     K = 9                   #Number of K-means clusters

  
    
#RG AND RTREES
if ((DO_RG==1)&(DO_BAYES==0)):
    max_iter = 25           #number of iterations to use in the K-means algorithm
    epsilon = .2           #epsilon for accuracy of K-means algorithm
    K_TRAIN = [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]                    #Number of K-means clusters
    K_TEST = 30
    K_ATTEMPTS = 20
    CLASSIFICATION_THRESHOLD = 0.4
#RGB AND RTREES
elif((DO_RG==0)&(DO_BAYES==0)):
    max_iter = 25           #number of iterations to use in the K-means algorithm
    epsilon = 1           #epsilon for accuracy of K-means algorithm
    K_TRAIN = 9                   #Number of K-means clusters
    #K_TRAIN = [4,6,8,9,4,5,9,5,9,8,7,8,9,13,4] 
    K_TRAIN = [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9] 
    K_TEST = 30
    K_ATTEMPTS = 20
    CLASSIFICATION_THRESHOLD = 0.4
#RGB AND BAYES
elif((DO_RG==0)&(DO_BAYES==1)):
    max_iter = 25           #number of iterations to use in the K-means algorithm
    epsilon = 1           #epsilon for accuracy of K-means algorithm
    K_TRAIN = [4,6,8,9,4,5,9,5,9,8,7,8,9,13,4]  
#     K_TRAIN = 9                   #Number of K-means clusters
    K_TEST = 30

    K_ATTEMPTS = 20
    CLASSIFICATION_THRESHOLD = 0.5
#RG AND BAYES   
elif((DO_RG==1)&(DO_BAYES==1)):    
    max_iter = 25           #number of iterations to use in the K-means algorithm
    epsilon = 1           #epsilon for accuracy of K-means algorithm
    #K_TRAIN = 5                   #Number of K-means clusters
    #K_TRAIN = [9,5,5,3,5,5,3,3,4,5,8,5,6,3,5] 
    K_TRAIN = [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5] 
    K_TEST = 10

    K_ATTEMPTS = 20
    CLASSIFICATION_THRESHOLD = 0.75
else:
    max_iter = 25           #number of iterations to use in the K-means algorithm
    epsilon = 1           #epsilon for accuracy of K-means algorithm
    K_TRAIN = 9                   #Number of K-means clusters
    K_TEST = 30

    K_ATTEMPTS = 20
    CLASSIFICATION_THRESHOLD = 0.5

IMG_TRAIN_PATH = '../face_training/'
IMG_TRAIN_GT_PATH = '../face_training_groundtruth/'
IMG_TEST_PATH = '../face_testing/'
IMG_TEST_GT_PATH = '../face_testing_groundtruth/'
IMG_TRAIN_RG_PATH = '../face_training_RG/'
IMG_TEST_RG_PATH = '../face_testing_RG/'

img_train = []
img_train_gt = []
ivec_train = []
fvec_train = []
center_train = []
label_train = []
kmean_train = []

img_test = []
img_test_orig = []
img_test_gt = []
ivec_test = []
fvec_test = []
center_test = []
label_test = []
kmean_test = []
center_test = []
predict_test = []

#RGB to RG conversion Function
def BGRtoGR(image):
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            d = float(image[x,y][0])+float(image[x,y][1])+float(image[x,y][2])
            if d != 0:
                G = image[x,y][1]/d
                R = image[x,y][2]/d
                B = 1 - G - R
                image[x,y][2] = 255*(R) 
                image[x,y][1] = 255*(G) 
                image[x,y][0] = 255*(B)
            else:
                image[x,y] = [0,0,0]            
    return #cvt_img

# ######################
# ## CREATE RG IMAGES ##
# ######################
# #Creating RG images takes too long every time so I do it once and then save the images
# #Read all training images into list
# 
# 
# i=0
# for filename in sorted(os.listdir(IMG_TRAIN_PATH)):
#     img = cv2.imread(os.path.join(IMG_TRAIN_PATH, filename))
#     if img is not None:
#         BGRtoGR(img)
#         cv2.imwrite(os.path.join(IMG_TRAIN_RG_PATH, filename),img)
#     i += 1
#       
# i = 0
# for filename in sorted(os.listdir(IMG_TEST_PATH)):
#     img = cv2.imread(os.path.join(IMG_TEST_PATH, filename))
#     if img is not None:
#         BGRtoGR(img)
#         cv2.imwrite(os.path.join(IMG_TEST_RG_PATH, filename),img)
#     i += 1    
#    
# sys.exit()

# 1. Read all training images into list
print "Reading Images..."
if DO_RG:
    for filename in os.listdir(IMG_TRAIN_RG_PATH):
        img = cv2.imread(os.path.join(IMG_TRAIN_RG_PATH, filename))
        if img is not None:
            img_train.append(img)
else:
    for filename in os.listdir(IMG_TRAIN_PATH):
        img = cv2.imread(os.path.join(IMG_TRAIN_PATH, filename))
        if img is not None:
            img_train.append(img)
                   

# 2. Read all training groundtruth images into list
for filename in os.listdir(IMG_TRAIN_GT_PATH):
    img = cv2.imread(os.path.join(IMG_TRAIN_GT_PATH, filename))
    if img is not None:
        img_thresh =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_train_gt.append(img_thresh)

# 3. READ IN TEST IMAGES
if DO_RG:
    #Read all test images into list
    for filename in os.listdir(IMG_TEST_RG_PATH):
        img = cv2.imread(os.path.join(IMG_TEST_RG_PATH, filename))
        if img is not None:
            img_test.append(img)
    #Also read in original images to use when displaying final results
    for filename in os.listdir(IMG_TEST_PATH):
        img = cv2.imread(os.path.join(IMG_TEST_PATH, filename))
        if img is not None:
            img_test_orig.append(img)
else:
    for filename in os.listdir(IMG_TEST_PATH):
        img = cv2.imread(os.path.join(IMG_TEST_PATH, filename))
        if img is not None:
            img_test.append(img)
            img_test_orig.append(img)

# 4. Read all test groundtruth images into list
for filename in os.listdir(IMG_TEST_GT_PATH):
    img = cv2.imread(os.path.join(IMG_TEST_GT_PATH, filename))
    if img is not None:
        img_thresh =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_test_gt.append(img_thresh)
 
     
############
# TRAINING #
############
#Reshape all images  
for x in img_train:
    ivec_train.append(x.reshape((-1,3)))
    
#Convert to float for K-means
for y in ivec_train:
    fvec_train.append(np.float32(y))

#Create K-Means Criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)


# ##################
# i=14
# for k in range(3,10):
#     ret, labels, centers = cv2.kmeans(fvec_train[i], k, criteria, K_ATTEMPTS, cv2.KMEANS_RANDOM_CENTERS)     #Get K-Means
#     #res = labels.reshape((img_train[i].shape))
#     cent = np.uint8(centers)                                                                #Convert centers from float to uint8
#     #center_train.append(cent)                                                               #Save Centers for later during training
#     res = cent[labels.flatten()]                                                            #Create new image based on center labels
#     res2 = res.reshape((img_train[i].shape))                                                #Reshape to original image shape
#      
#     xyz = labels.flatten()
#     xyz2 = xyz.reshape((img_train[i].shape[0],img_train[i].shape[1]))
#  
# #     center_train.append(centers)
#  
#     for c in centers:
#         center_train.append(c)
# #     if DEBUG_K_TRAIN:    
# #         cv2.imshow('Image: '+str(i),res2)
# #         cv2.waitKey()
# #     cv2.imshow('k: '+str(k), res2)
# #     cv2.waitKey()
#      
#     kmean_train.append(xyz2)
#  
#     # Count how many of each cluster are in image
#     region_count = [0]*k
#     overlap_count = [0]*k
#     class_label = [0]*k
#     for x in range(xyz2.shape[0]):
#         for y in range(xyz2.shape[1]):
#             region_count[xyz2[x,y]] += 1
#             #if (kmean_train[i][x,y] & img_train_gt[i][x,y]):
#             if img_train_gt[i][x,y]==255:
#                 overlap_count[xyz2[x,y]] += 1
#     #Check Strength of match:
#     a = np.array(overlap_count, dtype = np.float32)
#     b = np.array(region_count, dtype = np.float32)
#     ratio = a/b
#      
#     print "centers: "
#     print cent
#      
#     print np.array(a,np.integer)
#     print np.array(b,np.integer)
#     
#      
#     print xyz2[0,0]
#     print max(ratio)
#     if max(ratio)<CLASSIFICATION_THRESHOLD:
#         ratio[np.argmax(ratio,axis=0)] = 0.99
#     print "argmax:"
#     print np.argmax(ratio,axis=0)
#     
#     printratio = ["%.2f" % v for v in ratio]
#     print printratio
#      
#     for x,r in enumerate(ratio):
#         print x
#         print "r"+str(r)
#         if r > 0.98:
#             cent[x] = [0,200,200]
#         else:
#             if r > CLASSIFICATION_THRESHOLD:
#                 cent[x] = [200,0,0]
#      
#     chosen = cent[labels.flatten()]                                                            #Create new image based on center labels
#     chosen = chosen.reshape((img_train[i].shape))  
#     cv2.imshow('k: '+str(k), chosen)
#      
#      
# cv2.waitKey()    
# #################
#  
# cv2.destroyAllWindows()
# sys.exit()

#Perform K-Means on all training images
print "Perform K-Means Training..."
i = 0
for fvec in fvec_train:
    print "Image: " + str(i)
    ret, labels, centers = cv2.kmeans(fvec, K_TRAIN[i], criteria, K_ATTEMPTS, cv2.KMEANS_RANDOM_CENTERS)     #Get K-Means
    #res = labels.reshape((img_train[i].shape))
    cent = np.uint8(centers)                                                                #Convert centers from float to uint8
    #center_train.append(cent)                                                               #Save Centers for later during training
    res = cent[labels.flatten()]                                                            #Create new image based on center labels
    res2 = res.reshape((img_train[i].shape))                                                #Reshape to original image shape
    
    xyz = labels.flatten()
    xyz2 = xyz.reshape((img_train[i].shape[0],img_train[i].shape[1]))

#     center_train.append(centers)

    for c in centers:
        center_train.append(c)
#     if DEBUG_K_TRAIN:    
#         cv2.imshow('Image: '+str(i),res2)
#         cv2.waitKey()
#     if(i==2):
#         cv2.imshow('No matches?', res2)
#         cv2.waitKey()
          
    i += 1
    
    kmean_train.append(xyz2)
    

 
# CREATE MODEL
if DO_BAYES:
    bayes_model = cv2.NormalBayesClassifier()
else:
    rtrees_model = cv2.RTrees()


# PREPARE FOR CLASSIFICATION
for i in range(len(kmean_train)):
    # Count how many of each cluster are in image
    region_count = [0]*K_TRAIN[i]
    overlap_count = [0]*K_TRAIN[i]
    class_label = [0]*K_TRAIN[i]
    for x in range(kmean_train[i].shape[0]):
        for y in range(kmean_train[i].shape[1]):
            region_count[kmean_train[i][x,y]] += 1
            #if (kmean_train[i][x,y] & img_train_gt[i][x,y]):
            if img_train_gt[i][x,y]==255:
                overlap_count[kmean_train[i][x,y]] += 1
    #Check Strength of match:
    a = np.array(overlap_count, dtype = np.float32)
    b = np.array(region_count, dtype = np.float32)
    ratio = a/b
    
#     print np.array(a,np.integer)
#     print np.array(b,np.integer)
#     print ratio
    
#     #RG often ends up with low overlap, for training images that have no positive matches, just take the maximum
#     if DO_RG:
#         if max(ratio)<CLASSIFICATION_THRESHOLD:
#             ratio[np.argmax(ratio,axis=0)] = 0.99
    
    x = 0
    for element in ratio:
        if (element >= CLASSIFICATION_THRESHOLD):
            class_label[x] = 1
            label_train.append(1)
        else:
            class_label[x] = 0
            label_train.append(0)
        x+=1
    
#     if DEBUG_K_TRAIN:    
#         cv2.imshow('Image: '+str(i),img_train[i])
#         cv2.waitKey()
    #class_train.append(class_label)    
    
#     print "Image Number: " + str(i+1)
#     print overlap_count
#     print region_count
#     print ratio
#     print class_label

if DEBUG_K_TRAIN:
    cv2.destroyAllWindows()
    sys.exit() 


center_train = np.array(center_train, np.float32)
label_train = np.array(label_train, np.integer)

if DO_BAYES:
    bayes_model.train(center_train, label_train)
else:

    params = dict(max_depth=20,nactive_vars=3,max_num_trees_in_the_forest=3)

    rtrees_model.train(center_train, cv2.CV_ROW_SAMPLE, label_train, params = params)

###########
# TESTING #
###########
#Reshape all images  
for x in img_test:
    ivec_test.append(x.reshape((-1,3)))

#Convert to float for K-means
for y in ivec_test:
    fvec_test.append(np.float32(y))

#Create K-Means Criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)

#Perform K-Means on all training images
print "Perform K-means on Test Images..."
i = 0
for fvec in fvec_test:
    print "Image: " + str(i)
    ret, labels, centers = cv2.kmeans(fvec, K_TEST, criteria, K_ATTEMPTS, cv2.KMEANS_RANDOM_CENTERS)     #Get K-Means
    #res = labels.reshape((img_train[i].shape))
    cent = np.uint8(centers)                                                                #Convert centers from float to uint8
    #center_train.append(cent)                                                               #Save Centers for later during training
    res = cent[labels.flatten()]                                                            #Create new image based on center labels
    res2 = res.reshape((img_test[i].shape))                                                #Reshape to original image shape
    
    xyz = labels.flatten()
    xyz2 = xyz.reshape((img_test[i].shape[0],img_test[i].shape[1]))
    
    #Perform prediction for each found center
    for c in centers:
        c = np.reshape(c, (-1,3))
#         if DO_RG:
#             c = np.reshape(c, (-1,2))
#         else:
#             c = np.reshape(c, (-1,3))
        if DO_BAYES:
            prediction, _ = bayes_model.predict(np.array(c, np.float32))
        else:
            prediction = rtrees_model.predict(np.array(c, np.float32))
        if (prediction):
            predict_test.append(1)   
        else:
            predict_test.append(0)

    center_test.append(centers)
#     cv2.imshow('Image: '+str(i),res2)
#     cv2.waitKey()
    i += 1
    
    kmean_test.append(xyz2)

#print "label train:"
#print np.reshape(label_train,(15,-1))
predict_test = np.reshape(predict_test,(-1,K_TEST))
# print "predict test:"
# print predict_test
# print len(predict_test)
center_test = np.array(center_test)
# #print center_test
# print kmean_test[0]

print "Check Jaccard Score..."
for i in range(len(kmean_test)):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    
    for x in range(kmean_test[i].shape[0]):
        for y in range(kmean_test[i].shape[1]):
            if(predict_test[i][kmean_test[i][x,y]] == 1):
                #Should be skin
                #img_test_orig[i][x,y] = img_test_orig[i][x,y]#[255,0,0]
                img_test_orig[i][x,y][1] = 255
                img_test_orig[i][x,y][2] = 255
                #Count True Positives and False Positives
                if(img_test_gt[i][x,y] == 255):
                    TP += 1.0
                else:
                    FP += 1.0                
            else:
                #img_test_orig[i][x,y] = [0,0,0]
                #Not Skin
#                 img_test_orig[i][x,y] = [0,0,0]
                #Count True Positives and False Negatives
                if(img_test_gt[i][x,y] == 255):
                    FN += 1.0
#                 else:
#                     TP += 1.0
            
            
    #print jaccard(img_test_orig[i], img_test_gt[i])    
    #print "Jaccard - TP: " + str(TP) + " FP: " + str(FP) + " FN: " +str(FN) + " Result: " +str(float(TP/(TP+FP+FN)))    
    jac = float(TP/(TP+FP+FN))
    print jac
    cv2.imshow(str(jac), img_test_orig[i])

print "-- RUN TIME: %s seconds --" % (time.time() - start_time)
cv2.waitKey()




# print img_test[1][1,1]
# print img_test[1][1,1][0]
# print img_test[1][1,1][1]
# print img_test[1][1,1][2]

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



