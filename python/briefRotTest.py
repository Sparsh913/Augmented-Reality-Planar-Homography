import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
import scipy
from matplotlib import pyplot as plt

opts = get_opts()
#Q2.1.6
#Read the image and convert to grayscale, if necessary
img = cv2.imread('../data/cv_cover.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist = []
for i in range(1,36):
	#Rotate Image
	img_rot = scipy.ndimage.rotate(img, i*10, mode = 'constant')
	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(img, img_rot, opts)
	#Update histogram
	hist.append(matches.shape[0])
	#pass # comment out when code is ready
print(len(hist))
#x = np.linspace(10,350, 35)
#plt.bar(x, hist)

#Display histogram
plt.hist(hist, bins = 10)
plt.ylabel('Number of matches')
plt.show()
