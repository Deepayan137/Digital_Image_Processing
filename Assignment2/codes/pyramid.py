import cv2
import os
import sys
import pdb
import argparse
import numpy as np 
from skimage.exposure import rescale_intensity
sys.path.insert(0, '../smoothing/')
from filters import gaussian_filter
ap = argparse.ArgumentParser()
ap.add_argument("-I1", "--image", required=True, help="Path to the image")
ap.add_argument("-s", "--scale", type=float, default=2, help="scale factor size")
ap.add_argument("-l", "--levels", type=float, default=3, help="number of levels")
args = vars(ap.parse_args())


def pyramid(image, scale, levels):
	h, w = image.shape[0], image.shape[1]
	aspect_ratio = w/h
	min_size = (50,50)
	l=0
	#yield image 
	while True:
		#image = gaussian_filter(image, 3, 1)
		prior = image
		image = cv2.GaussianBlur(image,(5,5),1)
		width = int(image.shape[1]/scale)
		height = int(width/aspect_ratio)
		image = cv2.resize(image, (width, height))
		upsampled_image = np.zeros((prior.shape[0],prior.shape[1],prior.shape[2]),dtype='double')

		for z in range(0,image.shape[2]):
			for x in range(0,image.shape[1]):
				for y in range(0,image.shape[0]):
					upsampled_image[2*y,2*x,z] = image[y,x,z];

		upsampled_image = 4*cv2.GaussianBlur(upsampled_image,(5,5),1)
		lap_image = prior - upsampled_image
		lap_image = lap_image/255
		l+=1
		if width<min_size[1] or height<min_size[1] or l==levels+1:
			break
		yield image, lap_image
		
	
if __name__ == '__main__':
	image = cv2.imread(args["image"])
	image = np.array(image,dtype='double')
	ht, wd = image.shape[0], image.shape[1]
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	for (i, (resized, laplacian)) in enumerate(pyramid(image, scale=args["scale"], levels=args["levels"])): 
		cv2.imshow('orignal', image.astype("uint8"))
		cv2.imshow("Gaussian {}".format(i + 1), resized.astype('uint8'))
		cv2.imshow("Laplacian {}".format(i + 1), laplacian)
	cv2.waitKey(-1)
		#cv2.destroyAllWindows()