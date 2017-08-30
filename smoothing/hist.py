import os
import sys
import pdb
import cv2
import numpy as np 
from matplotlib import pyplot as plt
from skimage.exposure import rescale_intensity
from filters import median_filter
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--i", required=True, type=str)
argparser.add_argument("--t",  type=str)
argparser.add_argument("--method", required=True, type=str)
args = argparser.parse_args()

def imhist(image):
	hist = [0.0]*256
	height, width = image.shape[0], image.shape[1]
	#pdb.set_trace()
	for i in range(height):
		for k in range(width):
			hist[int(image[i,k])] +=1
	return np.array(hist)

def cumsum(hist):
	cumulative = np.array([sum(hist[:i+1]) for i in range(len(hist))])
	return (255.0 *cumulative/cumulative[-1])

def plot_hist(hist, cumhist):
	f = plt.figure(1, figsize=(20,10))
	plt.plot(list(range(len(cumhist))), cumhist, label='cumulative-histogram')
	#plt.plot(hist, list(range(len(hist))), '--', label='inverted-cumulative histogram')
	plt.legend()
	f.show()
	f.savefig('histeq_cumulative.png')
	plt.clf()
	g = plt.figure(2, figsize=(20, 10))
	plt.bar(list(range(len(hist))), hist, label='histogram')
	plt.legend()
	g.show()
	g.savefig('histeq_histogram.png')


def histeq(image):
	hist = imhist(image)
	cumulative= cumsum(hist)
	height, width = image.shape[0], image.shape[1]
	image_new = np.zeros(image.shape)
	for i in range(height):
		for k in range(width):
			image_new[i,k] = cumulative[image[i, k]]
	return image_new

def inverse_cumsum(cumulative):
	inv_cumsum = dict(list(zip(list(cumulative),list(range(256)))))
	return inv_cumsum
def closest(element, keys):
	keys_list = [item for item in keys]
	closest = [abs(element- item) for item in keys_list]
	#pdb.set_trace()
	#closest = min(range(len(keys_list)), key=lambda i: abs(keys_list[i]-element))
	index = np.argmin(np.array(closest))
	return keys_list[index]

def hist_matching(source, target):
	height, width = source.shape[0], source.shape[1]
	image_new = np.zeros(source.shape)
	source_hist = imhist(source)
	source_cumulative = cumsum(source_hist)
	target_hist = imhist(target)
	target_cumulative = cumsum(target_hist)
	inverse_target_cumulative = inverse_cumsum(target_hist)
	for i in range(height):
		print('iter[%d]/[%d]'%(i, height))
		for k in range(width):
			key = closest(source[i,k], inverse_target_cumulative.keys())
			image_new[i,k] = inverse_target_cumulative[key]
	return image_new

if __name__ == '__main__':
	source_image_path = args.i
	source_image = cv2.imread(source_image_path)
	source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
	
	if args.method == 'matching':
		target_image_path = args.t
		target_image = cv2.imread(target_image_path)
		target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
		new_source = hist_matching(source_gray, target_gray)
		output = rescale_intensity(new_source, in_range=(0, 255))
		new_source = (output*255).astype("uint8")
		cv2.imshow('orignal', source_gray.astype("uint8"))
		cv2.imshow('histogram_matched',new_source.astype("uint8"))
		cv2.waitKey(0)
	if args.method == 'equalize':
		new_source = histeq(source_gray)
		output = rescale_intensity(new_source, in_range=(0, 255))
		new_source = (output*255).astype("uint8")
		cv2.imshow('orignal', source_gray.astype("uint8"))
		cv2.imshow('histogram_equalized',new_source.astype("uint8"))
		cv2.waitKey(0)
	cv2.destroyAllWindows()