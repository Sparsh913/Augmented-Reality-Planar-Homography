import numpy as np
import cv2
#Import necessary functions
from PIL import Image
import imageio
from matchPics import matchPics
import planarH
from loadVid import loadVid
from opts import get_opts

ar_vid_path='../data/ar_source.mov'
book_vid_path='../data/book.mov'

# Loading Videos
ar = loadVid(ar_vid_path)
book = loadVid(book_vid_path)

diff_frame = book.shape[0] - ar.shape[0]
restart_frame = np.zeros((diff_frame, ar.shape[1], ar.shape[2], ar.shape[3]))
for i in range(diff_frame):
    # loop = i % ar.shape[0]
    restart_frame[i,:,:,:] = ar[i,:,:,:]

ar = np.concatenate((ar, restart_frame), axis=0)

#Write script for Q3.1
opts = get_opts()
ar_locs1 = []
ar_locs2 = []
ar_matches = []
ar_H2to1 = []
cv_cover = cv2.imread('../data/cv_cover.jpg')
ls_comp = []

for i in range(book.shape[0]):
    print('i :{} and book :{}'.format(i,book[i,:,:,:].shape))
    matches, locs1, locs2 = matchPics(book[i,:,:,:],cv_cover,opts)
    locs1[:,[0,1]] = locs1[:,[1,0]]
    locs2[:,[0,1]] = locs2[:,[1,0]]
    H2to1, inliers = planarH.computeH_ransac(locs1[matches[:,0]], locs2[matches[:,1]], opts)
    asp_r = cv_cover.shape[1]/cv_cover.shape[0] # Aspect ratio is width / height

    ar2bookcover = ar[i,:,:,:] # Extracting the ith frame. Now we can continue working as if we're working on an image
    ar2bookcover = ar2bookcover[44:-44,:] # Review once

    H, W, C = ar2bookcover.shape
    w = int(W/2)# Slicing the image from the middle column for width manipulation ahead
    width_ar2bookcover = H*asp_r # We'll use the Height of the region of interest in the frame in ar to be same
    ar2bookcover = ar2bookcover[:,int(w-width_ar2bookcover/2):int(w+width_ar2bookcover/2)]
    d = (cv_cover.shape[1], cv_cover.shape[0])
    ar2bookcover = cv2.resize(ar2bookcover, d)

    warp_img = planarH.compositeH(H2to1, ar2bookcover, book[i,:,:,:])
    ls_comp.append(warp_img)

def vid_release(frames, fps, save_path = "../data/ar.avi"):
    form = "XVID"
    fourcc = cv2.VideoWriter_fourcc(*form)
    s = frames[0].shape[1], frames[0].shape[0]
    vid = cv2.VideoWriter(save_path, fourcc, float(fps), s, isColor=True)
    for frame in frames:
        vid.write(np.uint8(frame))
    vid.release()
    return vid

vid_release(ls_comp, fps = 25)