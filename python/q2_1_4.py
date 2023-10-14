import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts

import planarH

opts = get_opts()

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')


matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)
#print("Shape of matches", matches.shape)
#display matched features
#n = matches.shape[0]
#x1 = np.zeros((n,2))
#x2 = np.zeros((n,2))
#for i in range(n):
#    idx1, idx2 = matches[i,0], matches[i,1]
#    x1[i,:] = locs1[idx1,:][::-1]
#    x2[i,:] = locs2[idx2,:][::-1]
#print("Shape of x1", x1.shape)
# matches, locs1, locs2 = matchPics(cv_cover, cv_cover, opts)
# locs1[:,[0,1]] = locs1[:,[1,0]]
# locs2[:,[0,1]] = locs2[:,[1,0]]
# H2to1, inliers = planarH.computeH_ransac(locs1[matches[:,0]], locs2[matches[:,1]], opts)

plotMatches(cv_cover, cv_desk, matches, locs1, locs2)
# plotMatches(cv_cover, cv_cover, matches, locs1, locs2) # Just for testing
# Testing ComputeH
#print(locs1)
#print(locs2)
#H2to1, inlier = planarH.computeH_ransac(x1, x2, opts)
# print(H2to1)
# print(inliers)