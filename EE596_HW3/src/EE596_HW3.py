'../Images/a2.jpg''''
Created on Nov 3, 2015

@author: Dan Sweet

EE596 Homework #3
'''

import cv2
import numpy as np
import os
import sys

# img1_in = cv2.imread('../Images/d2.png')          # queryImage
# img2_in = cv2.imread('../Images/d1.png') # trainImage

img1_in = cv2.imread('../Images/a2.jpg')          # queryImage
img2_in = cv2.imread('../Images/a1.jpg') # trainImage

# img1_in = cv2.imread('../Images/b2.jpg')          # queryImage
# img2_in = cv2.imread('../Images/b1.jpg') # trainImage
# 
# img1_in = cv2.imread('../Images/C2a.jpg')          # queryImage
# img2_in = cv2.imread('../Images/C1a.jpg') # trainImage
# minHessian = 180

img1 = cv2.cvtColor(img1_in,cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_in,cv2.COLOR_BGR2GRAY)

minHessian = 400
surf = cv2.SURF(minHessian)
kp1, des1 = surf.detectAndCompute(img1,None)
kp2, des2 = surf.detectAndCompute(img2,None)

# sift = cv2.SIFT()
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)

cv2.imshow("Right Image",cv2.drawKeypoints(img1_in,kp1))
cv2.imshow("Left Image",cv2.drawKeypoints(img2_in,kp2))


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.match(des1,des2)

src=[]
dst=[]
for m in matches:
    src.append(kp1[m.queryIdx].pt)
    dst.append(kp2[m.trainIdx].pt)



M, mask = cv2.findHomography(np.array(src), np.array(dst), cv2.RANSAC)#,5.0)
print M
im1 = img1_in
im2 = img2_in
 
warped = cv2.warpPerspective(im1, M, (im1.shape[1] + im2.shape[1], im1.shape[0]))
cv2.imshow("results1",warped)
cv2.waitKey()
warped[0:im2.shape[0], 0:im2.shape[1], :] = im2
cv2.imshow("results",warped)
cv2.waitKey()

print "kp_a1 len: "+str(len(kp1))
print "kp_a2 len: "+str(len(kp2))
print "Flann Match Length: " +str(len(matches))

# #filename = '../Images/a1.jpg'
# filename = '../Images/a2.jpg'
# # filename = '../Images/b1.jpg'
# # filename = '../Images/b2.jpg'
# # filename = '../Images/capri.jpg'
# 
# Input_Img = []
# Gray_Img = []
# kp_surf = []
# kp_sift = []
# des_surf = []
# des_sift = []
# 
# print "Reading Images..."
# 
# sift = cv2.SIFT() 
# for filename in os.listdir("../Images/"):
#     img = cv2.imread(os.path.join("../Images/", filename))
#     if img is not None:
#         Input_Img.append(img)
#         Gray_Img.append(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
#         
# 
# # - PART 1 on Capri Image - 
# #HARRIS
# gray_float = np.float32(Gray_Img[4])
# dst = cv2.cornerHarris(gray_float,2,3,0.04)
# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)
# # Threshold for an optimal value, it may vary depending on the image.
# 
# Input_Img[4][dst>0.02*dst.max()]=[0,0,255]
# 
# #Get SIFT and SURF point for all images
# sift = cv2.SIFT() 
# minHessian = 400
# surf = cv2.SURF(minHessian) 
# 
# kp_a1, des_a1 = surf.detectAndCompute(Gray_Img[0], None)
# kp_a2, des_a2 = surf.detectAndCompute(Gray_Img[1], None)
# 
# # for i in range(0,len(Input_Img)):
# #     print i
# #     # Collect and Save SIFT Values
# #     kp, des = sift.detectAndCompute(Gray_Img[i], None)
# #     kp_sift.append(kp)
# #     des_sift.append(des)
# #     # Collect and Save SURF Values
# #     kp, des = surf.detectAndCompute(Gray_Img[i], None)
# #     kp_surf.append(kp)
# #     des_surf.append(des)
#     
#     #DISPLAY
# #     cv2.imshow("sift: " +str(i),cv2.drawKeypoints(Gray_Img[i],kp_sift[i]))
# #     cv2.imshow("surf: " +str(i),cv2.drawKeypoints(Gray_Img[i],kp_surf[i]))
# 
# 
#     
# # kp_sift, des = sift.detectAndCompute(Gray_Img[4], None)
# # 
# # #SURF
# # 
# # 
# # kp_surf, des = surf.detectAndCompute(Gray_Img[4], None)
# # 
# # 
# # img_sift = cv2.drawKeypoints(Gray_Img[4],kp_sift)#,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# # img_surf = cv2.drawKeypoints(Gray_Img[4],kp_surf)
# # 
# # cv2.imshow("GRAY", Gray_Img[4])
# # cv2.imshow('HARRIS',img)
# # cv2.waitKey(0)
# # cv2.imshow('SIFT',img_sift)
# # cv2.waitKey(0)
# # cv2.imshow('SURF',img_surf)
# # cv2.waitKey(0)
# 
# 
# # print "SIFT Length: "+str(len(kp_sift))
# # print "SURF Length: "+str(len(kp_surf))
# # 
# 
# 
# 
# # # - PART 2 - Matching / Homography
# #FLANN
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)
# flann = cv2.FlannBasedMatcher(index_params,search_params)
# matches = flann.match(des_a1,des_a2)
# #matches = flann.knnMatch(des_a2,des_a1,k=2)
# 
# print "kp_a1 len: "+str(len(kp_a1))
# print "kp_a2 len: "+str(len(kp_a2))
# print "Flann Match Length: " +str(len(matches))
# 
# # 
# # good = []
# # for m,n in matches:
# #     if m.distance < 0.7*n.distance:
# #         good.append(m)
# 
# src = []
# dst = []
# 
# 
# src_pts = np.float32([ kp_a1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
# dst_pts = np.float32([ kp_a2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
# src_kp = [ kp_a1[m.queryIdx] for m in matches ]
# 
# print "kp_a1 len: "+str(len(kp_a1))
# print "mathes len: "+str(len(src_pts))
# cv2.imshow("jlkj",cv2.drawKeypoints(Gray_Img[0],src_kp))#,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.waitKey()
# cv2.imshow("orig",cv2.drawKeypoints(Gray_Img[0],kp_a1))
# cv2.waitKey()
# print src_pts
# print dst_pts
# 
# # for m in matches:
# #     good.append(m)
#     
# #     
# # src_pts = np.float32([ kp_a1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
# # 
# # good = []
# # for m,n in matches:
# #     if m.distance < 0.7*n.distance:
# #         good.append(m)
# 
# # src_pts = np.float32([ kp_a1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
# # dst_pts = np.float32([ kp_a2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
# # 
# # print src_pts[0]
# 
# M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)#,5.0)
# print M
# print mask
# matchesMask = mask.ravel().tolist()
# 
# im1 = Input_Img[1]
# im2 = Input_Img[0]
# 
# warped = cv2.warpPerspective(im1, M, (im1.shape[1] + im2.shape[1], im1.shape[0]))
# cv2.imshow("results1",warped)
# cv2.waitKey()
# warped[0:im2.shape[0], 0:im2.shape[1], :] = im2
# cv2.imshow("results",warped)
# 
# 
# # warped = cv2.warpPerspective(Input_Img[1], M, (Input_Img[1].shape[1] + Input_Img[0].shape[1], Input_Img[1].shape[0]))
# # cv2.imshow("results1",warped)
# # cv2.waitKey()
# # warped[0:Input_Img[0].shape[0], 0:Input_Img[0].shape[1], :] = Input_Img[0]
# # cv2.imshow("Image 0", Input_Img[0])
# # cv2.waitKey()
# # cv2.imshow("results",warped)
# 
# # h = Input_Img[0].shape[0]
# # w = Input_Img[0].shape[1]
# # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
# # dst = cv2.perspectiveTransform(pts,M)
# # 
# # 
# # 
# # img2 = cv2.polylines(Input_Img[1],[np.int32(dst)],True,255,3, cv2.LINE_AA)
# # 
# # cv2.imshow("matches",img2)
# 
# 
# 
# # matchesMask = [[0,0] for i in xrange(len(matches))]
# # 
# # # ratio test as per Lowe's paper
# # for i,(m,n) in enumerate(matches):
# #     if m.distance < 0.7*n.distance:
# #         matchesMask[i]=[1,0]
# # 
# # draw_params = dict(matchColor = (0,255,0),
# #                    singlePointColor = (255,0,0),
# #                    matchesMask = matchesMask,
# #                    flags = 0)
# # 
# # 
# # 
# # img3 = cv2.drawMatchesKnn(Input_Img[0],kp_a1,Input_Img[1],kp_a2,matches,None,**draw_params)
# # 
# # img3 = cv2.drawMatchesKnn(Input_Img[0],kp_a1,Input_Img[1],kp_a2,matches,None,**draw_params)
# # 
# # plt.imshow(img3,),plt.show()
# #  
# # print len(matches)
#  
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()