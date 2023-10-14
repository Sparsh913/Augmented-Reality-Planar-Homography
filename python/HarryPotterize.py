import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

#Import necessary functions
import planarH
from matchPics import matchPics


#Write script for Q2.2.4
opts = get_opts()
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

matches, locs1, locs2 = matchPics(cv_desk, cv_cover, opts)

locs1[:,[0,1]] = locs1[:,[1,0]]
locs2[:,[0,1]] = locs2[:,[1,0]]

H2to1, inliers = planarH.computeH_ransac(locs1[matches[:,0]], locs2[matches[:,1]], opts)
#H2to1 = planarH.computeH_norm(locs1[matches[:,0]], locs2[matches[:,1]])

rc_to_cr=(cv_cover.shape[1], cv_cover.shape[0])
hp_cover=cv2.resize(hp_cover, rc_to_cr)

composite_img = planarH.compositeH(H2to1, hp_cover, cv_desk)
print("Shape of composite image:",composite_img.shape)

cv2.imwrite('../data/harry_pot.jpg', composite_img)
print("No. of inliers", len(inliers[inliers==1]))