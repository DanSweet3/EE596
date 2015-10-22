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

start_time = time.time()

DO_RG = 1               #'0' = RG, '1' = RGB
DO_BAYES = 1            # 0' = BAYES , '1' = RANDOM TREES

SAVE_OUTPUT_IMAGES = 1  #'0' No output images saved, '1' Save output images

max_iter = 25           #number of iterations to use in the K-means algorithm (max tries to converge)
K_ATTEMPTS = 20         #Number of times to run K-means, best match is chosen

# TRAINING PARAMTERS - USE SLIGHTLY DIFFERENT VALUES FOR EACH VERSION (RGB/BAYES,RG/BAYES,RGB/RTRESS,RG/BAYES)
#RG AND RTREES
if ((DO_RG==1)&(DO_BAYES==0)): 
    epsilon = .2                                #accuracy of K-means algorithm
    K_TRAIN = 6                                 #K during training phase
    K_TEST = 30                                 #K during testing phase
    CLASSIFICATION_THRESHOLD = 0.4              
#RGB AND RTREES
elif((DO_RG==0)&(DO_BAYES==0)):
    epsilon = 1                                 #accuracy of K-means algorithm                             
    K_TRAIN = 9                                 #K during training phase
    K_TEST = 30                                 #K during testing phase
    CLASSIFICATION_THRESHOLD = 0.4
#RGB AND BAYES
elif((DO_RG==0)&(DO_BAYES==1)):
    epsilon = 1                                 #accuracy of K-means algorithm       
    K_TRAIN = 9                                 #K during training phase 
    K_TEST = 30                                 #K during testing phase
    CLASSIFICATION_THRESHOLD = 0.5
#RG AND BAYES   
elif((DO_RG==1)&(DO_BAYES==1)):    
    epsilon = 1                                 #accuracy of K-means algorithm
    K_TRAIN = 5                                 #K during training phase
    K_TEST = 10                                 #K during testing phase
    CLASSIFICATION_THRESHOLD = 0.75

#PATH CONSTANTS
IMG_TRAIN_PATH = '../face_training/'
IMG_TRAIN_GT_PATH = '../face_training_groundtruth/'
IMG_TEST_PATH = '../face_testing/'
IMG_TEST_GT_PATH = '../face_testing_groundtruth/'
IMG_TRAIN_RG_PATH = '../face_training_RG/'
IMG_TEST_RG_PATH = '../face_testing_RG/'
IMG_OUTPUT_PATH = '../Output/'

#LISTS FOR TRAINING
img_train = []
img_train_gt = []
ivec_train = []
fvec_train = []
center_train = []
label_train = []
kmean_train = []

#LISTS FOR TESTING
img_test = []
img_test_orig = []
img_test_gt = []
ivec_test = []
fvec_test = []
label_test = []
kmean_test = []
predict_test = []

###################################
## RGB to RG conversion Function ##
###################################
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
    return

######################################
## CREATE RG IMAGES FROM RGB IMAGES ##
######################################
#Creating RG images takes too long every time so I do it once and then save the images
#Read all training images into list
def Create_RG_Images():
    i=0
    for filename in sorted(os.listdir(IMG_TRAIN_PATH)):
        img = cv2.imread(os.path.join(IMG_TRAIN_PATH, filename))
        if img is not None:
            BGRtoGR(img)
            cv2.imwrite(os.path.join(IMG_TRAIN_RG_PATH, filename),img)
        i += 1
           
    i = 0
    for filename in sorted(os.listdir(IMG_TEST_PATH)):
        img = cv2.imread(os.path.join(IMG_TEST_PATH, filename))
        if img is not None:
            BGRtoGR(img)
            cv2.imwrite(os.path.join(IMG_TEST_RG_PATH, filename),img)
        i += 1    
    return

###########
## START ##
###########

# 1. Check to see if RG Images have been made (I only do this once and save the results, to improve run time)
if DO_RG:
    if (len(os.listdir(IMG_TRAIN_RG_PATH)) == 0) | (len(os.listdir(IMG_TEST_RG_PATH)) == 0):
        print "Creating RG Images..."
        Create_RG_Images()
        
# 2. Read all training images into list
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
                   
# 3. Read all training groundtruth images into list
for filename in os.listdir(IMG_TRAIN_GT_PATH):
    img = cv2.imread(os.path.join(IMG_TRAIN_GT_PATH, filename))
    if img is not None:
        img_thresh =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_train_gt.append(img_thresh)

# 4. READ IN TEST IMAGES
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

# 5. Read all test groundtruth images into list
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

#Perform K-Means on all training images
print "Perform K-Means Training..."
for (i,fvec) in enumerate(fvec_train):
    print "Image: " + str(i)
    ret, labels, centers = cv2.kmeans(fvec, K_TRAIN, criteria, K_ATTEMPTS, cv2.KMEANS_RANDOM_CENTERS)     #Get K-Means
    #Reshape and save labels
    labels = labels.flatten()
    labels = labels.reshape((img_train[i].shape[0],img_train[i].shape[1]))
    kmean_train.append(labels)
    #Save centers for use during training
    for c in centers:
        center_train.append(c)

# CREATE MODEL
if DO_BAYES:
    bayes_model = cv2.NormalBayesClassifier()
else:
    rtrees_model = cv2.RTrees()

# PREPARE FOR CLASSIFICATION (CHECK FOR 50% OVERLAP)
for i in range(len(kmean_train)):
    # Count how many of each cluster are in image
    region_count = [0]*K_TRAIN
    overlap_count = [0]*K_TRAIN
    for x in range(kmean_train[i].shape[0]):
        for y in range(kmean_train[i].shape[1]):
            region_count[kmean_train[i][x,y]] += 1
            if img_train_gt[i][x,y]==255:
                overlap_count[kmean_train[i][x,y]] += 1
    
    #Check Strength of match ratio:
    a = np.array(overlap_count, dtype = np.float32)
    b = np.array(region_count, dtype = np.float32)
    ratio = a/b

    for (x,element) in enumerate(ratio):
        if (element >= CLASSIFICATION_THRESHOLD):
            label_train.append(1)
        else:
            label_train.append(0)

center_train = np.array(center_train, np.float32)
label_train = np.array(label_train, np.integer)

#TRAIN THE MODEL
if DO_BAYES:
    bayes_model.train(center_train, label_train)
else: #RTREES
    params = dict(max_depth=20,nactive_vars=3,max_num_trees_in_the_forest=3)
    rtrees_model.train(center_train, cv2.CV_ROW_SAMPLE, label_train, params = params)

###########
# TESTING #
###########
#Reshape all test images  
for x in img_test:
    ivec_test.append(x.reshape((-1,3)))

#Convert to float for K-means
for y in ivec_test:
    fvec_test.append(np.float32(y))

#Perform K-Means on all training images
print "Perform K-means on Test Images..."

for (i,fvec) in enumerate(fvec_test):
    print "Image: " + str(i)
    ret, labels, centers = cv2.kmeans(fvec, K_TEST, criteria, K_ATTEMPTS, cv2.KMEANS_RANDOM_CENTERS)     #Get K-Means
    
    #Format and save the labels list
    labels = labels.flatten()
    labels = labels.reshape((img_test[i].shape[0],img_test[i].shape[1]))
    kmean_test.append(labels)
    
    #Perform prediction for each center
    for c in centers:
        c = np.reshape(c, (-1,3))
        if DO_BAYES:
            prediction, _ = bayes_model.predict(np.array(c, np.float32))
        else:
            prediction = rtrees_model.predict(np.array(c, np.float32))
        if (prediction):
            predict_test.append(1)   
        else:
            predict_test.append(0)

#Reformat Predictions and Centers
predict_test = np.reshape(predict_test,(-1,K_TEST))

#################
# CHECK RESULTS #
#################
print "Check Jaccard Score..."
for i in range(len(kmean_test)):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    
    for x in range(kmean_test[i].shape[0]):
        for y in range(kmean_test[i].shape[1]):
            if(predict_test[i][kmean_test[i][x,y]] == 1):
                #Should be skin (highlight yellow)
                img_test_orig[i][x,y][1] = 255
                img_test_orig[i][x,y][2] = 255
                #Count True Positives and False Positives
                if(img_test_gt[i][x,y] == 255):
                    TP += 1.0
                else:
                    FP += 1.0                
            else:
                #Not Skin, Count False Negatives
                if(img_test_gt[i][x,y] == 255):
                    FN += 1.0
    #Calculate Jaccard score and limit to 2 decimal places                
    jac = "%.3f" % (float(TP/(TP+FP+FN)))
    print jac
    cv2.imshow(str(jac), img_test_orig[i])
    
    if SAVE_OUTPUT_IMAGES:
        if ((DO_RG == 0)&(DO_BAYES == 0)):
            filename = "RGB_RTREES_"
        elif ((DO_RG == 0)&(DO_BAYES == 1)):
            filename = "RGB_BAYES_"
        elif ((DO_RG == 1)&(DO_BAYES == 0)):
            filename = "RG_RTREES_"
        else:
            filename = "RG_BAYES_"
        cv2.imwrite(os.path.join(IMG_OUTPUT_PATH, filename+str(i)+'.png'),img_test_orig[i])

print "-- RUN TIME: %s seconds --" % (time.time() - start_time)
cv2.waitKey()
cv2.destroyAllWindows()
