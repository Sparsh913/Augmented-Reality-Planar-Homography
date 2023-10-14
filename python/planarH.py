import numpy as np
import cv2
import random

import ipdb
st = ipdb.set_trace


def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
	assert(x1.shape[0] == x2.shape[0])
    #assert(x1.shape[1] == 2)
	n = x1.shape[0]
	u, v = x1[:,0].reshape(n,1), x1[:,1].reshape(n,1)
	x, y = x2[:,0].reshape(n,1), x2[:,1].reshape(n,1)

	r1 = np.concatenate((x, y, np.ones((n,1)), np.zeros((n,3)), -np.multiply(x,u), -np.multiply(y,u), -u), axis=1)
	r2 = np.concatenate((np.zeros((n,3)), x, y, np.ones((n,1)), -np.multiply(x,v), -np.multiply(y,v), -v), axis=1)
	A = np.concatenate((r1, r2), axis = 0)

	# Solving Ah = 0
	U, S, VT = np.linalg.svd(A, full_matrices=False)
	h = VT[-1,:]
	H2to1 = h.reshape(3,3)

	return H2to1


def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points
	mean_x1 = np.mean(x1,axis=0)
	mean_x2 = np.mean(x2,axis=0)
	# print("Shape of mean_x1", mean_x1.shape)

	#Shift the origin of the points to the centroid
	shift_x1 = x1 - mean_x1
	shift_x2 = x2 - mean_x2
	# print("Shape of shift_x1", shift_x1.shape)

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	x1_max = np.amax(abs(shift_x1), axis=0)
	x2_max = np.amax(abs(shift_x2), axis=0)
	# print("x1_max Shape", x1_max.shape)

	for i in range(2):
		if x1_max[i] == 0:
			x1_max[i] = 10**(-8)
		if x2_max[i] == 0:
			x2_max[i] = 10**(-8)
	
	norm_x1 = shift_x1/x1_max
	norm_x2 = shift_x2/x2_max
	# print("Shape of norm_x1", norm_x1.shape)

	#Similarity transform 1
	T1=np.array(([1/x1_max[0], 0, -mean_x1[0]/x1_max[0]],[0, 1/x1_max[1], -mean_x1[1]/x1_max[1]],[0,0,1]))
	#T1 = np.array([[1/x1_max[0]]])
	# print("Shape T1", T1.shape)
	#Similarity transform 2
	T2=np.array(([1/x2_max[0], 0, -mean_x2[0]/x2_max[0]],[0, 1/x2_max[1], -mean_x2[1]/x2_max[1]],[0,0,1]))

	#Compute homography
	H2to1_norm = computeH(norm_x1, norm_x2)

	#Denormalization
	H2to1 = np.linalg.inv(T1)@H2to1_norm@T2
	# H2to1 = cv2.findHomography(x1, x2)

	return H2to1




def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier
	
	assert(locs1.shape[0] == locs2.shape[0])
	n = locs1.shape[0]
	# print("Shape locs1", locs1.shape)
	inlier = np.zeros(max_iters)
	homographies = []
	s = 4 # At least 4 points are needed for homography
	for i in range(max_iters):
		try:
			rn = random.sample(range(1, n), s)
		except:
			st()
		# print("rn", rn)
		# Fit the model
		x1 = np.column_stack((locs1[rn[0],:], locs1[rn[1],:], locs1[rn[2],:], locs1[rn[3],:])).T
		x2 = np.column_stack((locs2[rn[0],:], locs2[rn[1],:], locs2[rn[2],:], locs2[rn[3],:])).T
		# print("Shape of x1", x1.shape) # (4,2)
		H = computeH_norm(x1, x2)
		homographies.append(H)
		homog_x2 = np.concatenate([locs2, np.ones((n,1))], axis = 1).T
		# print("Shape of homog_x2", homog_x2.shape) # (3,n)
		homog_x1 = H @ homog_x2
		# print("Shape of homog_x1", homog_x1.shape) # (3,n)
		locs1_one = np.concatenate([locs1, np.ones((n,1))], axis = 1).T
		# Convert the retrieved homogeneous coordinates back to cartesian coordinates for comparison
		cart_x1 = np.column_stack((homog_x1[0,:]/homog_x1[2,:], homog_x1[1,:]/homog_x1[2,:]))
		# cart_x1 = np.column_stack((homog_x1[0,:], homog_x1[1,:]))
		# print("Shape of cart_x1", cart_x1.shape) # (n,2)
		# Compare & compute the no. of inliers
		comp = np.linalg.norm((cart_x1 - locs1), axis=1)
		# homog_x1_new = homog_x1/homog_x1[-1,:]
		# comp = np.linalg.norm((homog_x1_new - locs1_one), axis=0)
		# print("Shape comp", comp.shape) # (n,)
		# print("Initial Values of comp", comp[:10])
		inlier[i] = len(comp[comp <= inlier_tol])
		# print("Shape inlier", inlier.shape)

	max_idx = np.argmax(inlier)
	bestH2to1 = homographies[max_idx]
	homog_x1 = bestH2to1 @ homog_x2
	cart_x1 = np.column_stack((homog_x1[0,:]/homog_x1[2,:], homog_x1[1,:]/homog_x1[2,:]))
	comp = np.linalg.norm((cart_x1 - locs1), axis=1)
	# print("Initial Values of comp outside the loop", comp[:10])
	inliers = np.where(comp <= inlier_tol, 1, 0)
	# print("Number of Matches after Ransac", len(inliers[inliers == 1]))
	# print("% Match", len(inliers[inliers == 1])*100/n)
	# print("Max no. of matches", inlier[max_idx])
	# print("Max no. of matches_", np.max(inlier))
	# print("Max_idx", max_idx)
	# print("Inlier", inlier)
	# print("Inliers", inliers)

	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	

	#Create mask of same size as template
	mask = np.ones((template.shape[0], template.shape[1], 3), dtype = np.uint8)

	#Warp mask by appropriate homography
	warp_mask = cv2.warpPerspective(mask, H2to1, (img.shape[1], img.shape[0]))
	cv2.imwrite('../data/warp_mask1.jpg', warp_mask)

	#Warp template by appropriate homography
	warp_template = cv2.warpPerspective(template, H2to1, (img.shape[1], img.shape[0]))
	cv2.imwrite('../data/warp_temp1.jpg', warp_template)

	iwm = np.where(warp_mask >= 1,0,1).astype(dtype=np.uint8)

	#Use mask to combine the warped template and the image
	img_cut = np.multiply(img, iwm)
	composite_img = img_cut + warp_template
	
	return composite_img


