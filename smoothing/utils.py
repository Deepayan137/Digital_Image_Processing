import os
import sys
import numpy as np 
import pdb

def padding(image, pad):
	height, width  = image.shape[0], image.shape[1]
	#pdb.set_trace()
	im_bg = np.zeros((int(height + 2*pad), int(width + 2*pad)), dtype="float")
	im_bg = (im_bg+1)*255
	im_bg[pad:pad+height, pad:pad+width] = (image)
	return im_bg, height, width

def convolution(image, kernel):
	
	kh, kw = kernel.shape[0], kernel.shape[1]
	pad = int((kw-1)/2)
	im_bg,h, w = padding(image, pad)
	result = np.zeros((h, w), dtype="float")
	for y in np.arange(pad, h+pad):
		for x in np.arange(pad, w+pad):
			#pdb.set_trace()
			roi = im_bg[y - pad:y + pad + 1, x - pad:x + pad + 1]
			k = (roi * kernel).sum()
			result[y - pad, x - pad] = k
	
	return result