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

def downsample(image, sigma):
	#	pdb.set_trace()
	h, w = image.shape[0], image.shape[1]
	aspect_ratio = w/h
	scale = 2
	min_size = (50,50)
	image = cv2.GaussianBlur(image,(5,5),sigma)
	
	width = int(image.shape[1]/scale)
	height = int(width/aspect_ratio)
	resized_image = cv2.resize(image, (width, height))
	return resized_image
	

def upsample(image, next_image):
	h, w = next_image.shape[0], next_image.shape[1]
	upsampled_image = np.zeros(next_image.shape,dtype='double')
	#print(upsampled_image.shape,image.shape)

	for z in range(0,image.shape[2]):
			for x in range(0,image.shape[1]):
				for y in range(0,image.shape[0]):
					upsampled_image[2*y,2*x,z] = image[y,x,z];

	upsampled_image = 4*cv2.GaussianBlur(upsampled_image,(5,5),1)
	return upsampled_image


I1 = np.array(cv2.imread(args["image1"]),dtype='double')/255
I2 = np.array(cv2.imread(args["image2"]),dtype='double')/255

mask = np.array(cv2.imread(args["mask"]),dtype='double')/255



A = I1.copy()
gpA = [A]
for i in range(5):
	#pdb.set_trace()
	#print(A.shape)

	if A.shape[0]>30 and A.shape[1]>30:
		A = downsample(A,i+1)
		gpA.append(A)
	else:
		break

B = I2.copy()
gpB = [B]
for i in range(5):
	#print(B.shape)
	if B.shape[0]>30 and B.shape[1]>30:
		B = downsample(B, i+1)
		gpB.append(B)
	else:
		break

M = mask.copy()
gpM = [M]
for i in range(5):
	#print(M.shape)
	if M.shape[0]>30 and M.shape[1]>30:
		M = downsample(M,i+1)
		
		gpM.append(M)
	else:
		break

lpA = [gpA[-1]]

for i in range(5, 0, -1):
	GE = upsample(gpA[i], gpA[i-1])
	
	L = (gpA[i-1]-GE)
	lpA.append(L)
lpB = [gpB[-1]]

for i in range(5, 0, -1):
	
	GE = upsample(gpB[i], gpB[i-1])
	L = (gpB[i-1]-GE)
	lpB.append(L)

LS = []

for la,lb,gm in zip(lpA,lpB,gpM[::-1]):
	gim = np.ones(gm.shape)
	
	gim = gim-gm
	#print(la.shape, gim.shape)
	la = np.multiply(la,gim)
	lb = np.multiply(lb,gm)
	ls = cv2.add(la,lb)
	LS.append(ls)
	
ls_ = LS[0]
l =[]

for i in range(1,5):
	

	ls_ = upsample(ls_, LS[i])
	ls_ = cv2.add(ls_, LS[i])
	l.append(ls_)

for i in range(len(l)):
	
	cv2.imshow("Blended  {}".format(i + 1),l[i])
	#cv2.imshow("Laplacian {}".format(i + 1), lpA[i])
cv2.waitKey(-1)
#cv2.imwrite('Pyramid_blending_3.jpg',ls_)