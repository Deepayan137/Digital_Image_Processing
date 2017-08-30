import numpy as np 
from math import e
import pdb
from utils import convolution, padding


def gaussian_kernel(image, n_dim, sigma):
	gauss_filter = np.zeros((n_dim, n_dim))
	normalization_constant = 1/(float(np.sqrt(2*np.pi))*2*sigma)
	pos = np.arange(-(n_dim//2), (n_dim//2)+1)
	for i in range(n_dim):
		for j in range(n_dim):
			gauss_filter[j,i] = np.exp(-((pos[i]**2 + pos[j]**2)/float(sigma**2)))
	kernel = gauss_filter*normalization_constant
	return kernel 


def gaussian_filter(image, n_dim, sigma):
	kernel = gaussian_kernel(image, n_dim, sigma)
	output = convolution(image, kernel)		
	return output

def median_filter(image, n_dim):
	
	pad = int((n_dim-1)/2)
	im_bg,h, w = padding(image, pad)
	result = np.zeros((h,w), dtype='float')
	for y in np.arange(pad, h+pad):
		for x in np.arange(pad, w+pad):
			roi = im_bg[y - pad:y + pad + 1, x - pad:x + pad + 1]
			k = np.median(roi)
			result[y-pad, x-pad] = k
	return result

def mean_filter(image, n_dim):
	
	pad = int((n_dim-1)/2)
	im_bg,h, w = padding(image, pad)
	result = np.zeros((h,w), dtype='float')
	for y in np.arange(pad, h+pad):
		for x in np.arange(pad, w+pad):
			roi = im_bg[y - pad:y + pad + 1, x - pad:x + pad + 1]
			k = np.mean(roi)
			result[y-pad, x-pad] = k
	return result

def highboost_filter(image, n_dim, lam):
	mean_image = mean_filter(image, n_dim)
	edges = image - mean_image
	output = image + lam*edges
	return(output)


def bilateral_filter(image, n_dim, sigma, sigma_range):
	bilateral_filter = np.zeros((n_dim, n_dim))
	pad = int((n_dim-1)/2)
	e=''
	im_bg,h, w = padding(image, pad)
	gaussian_filter = gaussian_kernel(image, n_dim, sigma)
	result = np.zeros((h,w), dtype='float')	
	try:	
		for y in np.arange(pad, h+pad):
			for x in np.arange(pad, w+pad):
				roi = im_bg[y - pad:y + pad + 1, x - pad:x + pad + 1]
				#pdb.set_trace()
				range_filter = np.exp(-((roi-image[y, x])**2)/float(2*sigma_range**2))
				
				bilateral_filter = np.multiply(range_filter, gaussian_filter)
				bilateral_filter = bilateral_filter/np.sum(np.sum(bilateral_filter, axis=0))
	except Exception as e:
		print ('error')
		print(e)
		
	#pdb.set_trace()

	output = convolution(image, bilateral_filter)
	return output