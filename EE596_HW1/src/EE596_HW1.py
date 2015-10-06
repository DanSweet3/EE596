'''
Created on Oct 1, 2015

@author: Dan Sweet

EE596 Homework 1
'''
import cv2
import numpy as np

BACKGROUND_COLOR = (0,0,0)

#Function to process a passed image with a given binary threshold, close size and open size.
# Performs the following:
#  1. Performs binary threshold given image with passed binary_threshold parameter
#  2. Closes the image with passed close_size parameter. This filters small black dots out of each major organ
#  3. Opens the image with passed open_size parameter. This leaves only large blocks (organs) visible
#  4. Finds contours of image, keeping track of hierarchy to not lose nested contours. 

def ProcessImage(image, binary_threshold, close_size, open_size):
    _, img_thresh = cv2.threshold(image, binary_threshold, 250, cv2.THRESH_BINARY)    #Perform binary threshold to get black and white
    cv2.imshow('GRAY', image)                                            #Show Grayscale
    cv2.imshow('THRESHOLD', img_thresh)                                     #Show binary threshold image
    
    # MORPHING
    # First close the image to get rid of black dots within larger shapes
    circ_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(close_size,close_size))  
    img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, circ_mask)     
    cv2.imshow('CLOSED', img_morph)    
    
    # Then open to get rid of white elements that are too small to be significant
    circ_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(open_size,open_size))
    img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_OPEN, circ_mask)    
    cv2.imshow('OPENED', img_morph)
       
    #CONTOURS
    img_contours = np.zeros((image.shape[0], image.shape[1],3), np.uint8)     #Create empty image
    #Get contours, with hierarchy so I don't lose nested (child) contours
    contours, heirarchy = cv2.findContours(img_morph, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) #Get contours
    #Add each contour with a random color 
    for contour_idx in range(len(contours)):
        #if not a child contour, draw with random color
        if heirarchy[0][contour_idx,3] == -1:
            cv2.drawContours(img_contours, contours, contour_idx, np.random.randint(0,255,3), -1)
        #if a child contour, just draw as background color
        else:
            cv2.drawContours(img_contours, contours, contour_idx, BACKGROUND_COLOR, -1)
    cv2.imshow('CONTOURS', img_contours)                                    #Show contoured/colored output
    cv2.waitKey(0)
    
    return

#Call Process Image on the three pictures, with slightly different parameters for each
ProcessImage(cv2.imread('A:/UW/EE596/kidney.png', 0), 130, 3, 12);
ProcessImage(cv2.imread('A:/UW/EE596/e030.png', 0),   132, 3, 14);
ProcessImage(cv2.imread('A:/UW/EE596/g006.png', 0),   130, 2, 10);
