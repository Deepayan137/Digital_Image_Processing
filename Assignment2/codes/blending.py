import cv2
import os
import argparse
import pdb
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("-I1", "--image1", required=True, help="Path to the image")
ap.add_argument("-I2", "--image2", required=True, help="Path to the image")
ap.add_argument("--mask", required=True, help="Path to the mask")
args = vars(ap.parse_args())

def downsample(image):
	#	pdb.set_trace()
	h, w = image.shape[0], image.shape[1]
	aspect_ratio = w/h
	scale = 2
	min_size = (50,50)
	image = cv2.GaussianBlur(image,(5,5),2)
	if (w < min_size[1] or h < min_size[0]) != True:
		width = int(image.shape[1]/scale)
		height = int(width/aspect_ratio)
		resized_image = cv2.resize(image, (width, height))
		return resized_image
	else:
		return image

def upsample(image, next_image):
	h, w = next_image.shape[0], next_image.shape[1]
	resized_image = cv2.resize(image, (w,h))
	return resized_image


I1 = cv2.imread(args["image1"])
I2 = cv2.imread(args["image2"])
mask = cv2.imread(args["mask"])
mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
inv_mask = cv2.bitwise_not(mask)

I1 = cv2.bitwise_and(I1, I1, mask = mask)
I2 = cv2.bitwise_and(I2,I2, mask = mask)
cv2.imshow('masked',I2)
A = I1.copy()
gpA = [A]
for i in range(4):
	A = downsample(A)
	gpA.append(A)

B = I2.copy()
gpB = [B]
for i in range(4):
	B = downsample(B)
	gpB.append(B)

M = mask.copy()


lpA = cv2.add(gpA[-1], cv2.resize(M,gpA[-1].shape))
for i in range(3, 0, -1):
	GE = upsample(gpA[i], gpA[i-1])
	GM = upsample(gpM[i], gpM[i-1])
	#pdb.set_trace()

	L = np.subtract(gpA[i-1],GE)
	lpA.append(L)

lpB = [gpB[-1]]
for i in range(3, 0, -1):
	GE = upsample(gpB[i], gpB[i-1])
	L = np.subtract(gpB[i-1],GE)
	lpB.append(L)

LS = []
for la,lb in zip(lpA,lpB):
	rows,cols,dpt = la.shape
	#pdb.set_trace()	
	ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
	#ls = np.hstack((la, lb))
	LS.append(ls)

ls_ = LS[0]
l =[]
for i in range(1,4):
	#pdb.set_trace()
	ls_ = upsample(ls_, LS[i])
	ls_ = np.add(ls_, LS[i])
	l.append(ls_)
for i in range(len(l)):

	cv2.imshow('image', )

cv2.waitKey(0)
#cv2.imwrite('Pyramid_blending.jpg',ls_)