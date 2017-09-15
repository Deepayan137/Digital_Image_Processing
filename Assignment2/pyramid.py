import cv2
import os
import sys
import pdb
import argparse
from skimage.exposure import rescale_intensity
sys.path.insert(0, '../smoothing/')
from filters import gaussian_filter
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-s", "--scale", type=float, default=1.5, help="scale factor size")
args = vars(ap.parse_args())


def pyramid(image, scale):
	h, w = image.shape[0], image.shape[1]
	aspect_ratio = w/h
	min_size = (50,50)
	#yield image 
	while True:
		#image = gaussian_filter(image, 3, 1)
		prior = image
		image = cv2.GaussianBlur(image,(5,5),2)
		width = int(image.shape[1]/scale)
		height = int(width/aspect_ratio)
		#pdb.set_trace()
		image = cv2.resize(image, (width, height))
		upsampled_image = cv2.resize(image, (prior.shape[1], prior.shape[0]))
		lap_image = prior - upsampled_image
		#image = rescale_intensity(image, in_range=(0, 255))
		#image = (image*255).astype("uint8")
		if width<min_size[1] or height<min_size[1]:
			break
		yield image, lap_image
		
	
if __name__ == '__main__':
	image = cv2.imread(args["image"])
	ht, wd = image.shape[0], image.shape[1]
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	for (i, (resized, laplacian)) in enumerate(pyramid(image, scale=args["scale"])):
	# show the resized image
		#resized = cv2.resize(resized, (wd, ht))
		#cv2.namedWindow("Layer {}".format(i + 1), cv2.WINDOW_NORMAL) 
		cv2.imshow("Gaussian {}".format(i + 1), resized)
		cv2.imshow("Laplacian {}".format(i + 1), laplacian)
		cv2.waitKey(-1)
		#cv2.destroyAllWindows()