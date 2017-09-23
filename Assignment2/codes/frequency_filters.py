import cv2
from matplotlib import pyplot as plt
import argparse
import pdb
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("-I1", "--image1", required=True, help="Path to the image")
ap.add_argument("-m", "--method", required=True, help="filter to use")
args = vars(ap.parse_args())

method = args['method']
img = cv2.imread(args['image1'], 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

h,w = fshift.shape[0], fshift.shape[1]
center = (h//2, w//2)
#pdb.set_trace()
Do = 30
H = np.zeros(fshift.shape)
D = np.zeros(fshift.shape)
def lowpass(image):
	for u in range(fshift.shape[0]):
		for v in range(fshift.shape[1]):
			D[u,v] = np.abs(np.sqrt((u-center[0])**2 + (v-center[1])**2))
			#pdb.set_trace()
			if D[u,v] <= Do:
				H[u,v] = 1
	return H


def laplacian(image):
	for u in range(fshift.shape[0]):
		for v in range(fshift.shape[1]):
			D[u,v] = np.abs(np.sqrt((u-center[0])**2 + (v-center[1])**2))
			H[u,v] = 4*(np.pi*D[u,v])**2
	return H

def butterworth(image, n=5):
	
	for u in range(fshift.shape[0]):
		for v in range(fshift.shape[1]):
			D[u,v] = np.abs(np.sqrt((u-center[0])**2 + (v-center[1])**2))
			H[u,v] = 1/(1+ (D[u,v]/Do)**2*n)
	return H

def gaussian(image):
	
	for u in range(fshift.shape[0]):
		for v in range(fshift.shape[1]):
			D[u,v] = np.abs(np.sqrt((u-center[0])**2 + (v-center[1])**2))
			factor = (D[u,v]/(np.sqrt(2)*Do))**2
			H[u,v] = np.exp(-(factor))
	return H


def highpass(image):
	H = lowpass(image)
	H_ = 1 - H
	return H_

def highpass_butterworth(image, n=2):
	H = butterworth(image, n)
	H_ = 1 - H
	return H_

def highpass_gaussian(image):
	H = gaussian(image)
	H_ = 1 - H
	return H_


def notch1(image):
	x = [136, 171, 150, 184]
	y = [78, 178]
	Do = 30
	H = np.ones(image.shape)
	for u in range(fshift.shape[0]):
		for v in range(fshift.shape[1]):

			D[u,v] = np.abs(np.sqrt((u-78)**2 + (v-136)**2))
			if D[u,v] <= Do:
				H[u,v] = 0

			D[u,v] = np.abs(np.sqrt((u-78)**2 + (v-171)**2))
			if D[u,v] <= Do:
				H[u,v] = 0

			D[u,v] = np.abs(np.sqrt((u-178)**2 + (v-150)**2))
			if D[u,v] <= Do:
				H[u,v] = 0

			D[u,v] = np.abs(np.sqrt((u-178)**2 + (v-184)**2))
			if D[u,v] <= Do:
				H[u,v] = 0
				
	return H 


def notch2(image):
	x = [23,108, 44, 87]
	y = [90, 41, 53, 77]
	Do = 10
	H = np.ones(image.shape)
	for u in range(fshift.shape[0]):
		for v in range(fshift.shape[1]):

			D[u,v] = np.abs(np.sqrt((u-90)**2 + (v-23)**2))
			if D[u,v] <= Do:
				H[u,v] = 0

			D[u,v] = np.abs(np.sqrt((u-41)**2 + (v-108)**2))
			if D[u,v] <= Do:
				H[u,v] = 0

			D[u,v] = np.abs(np.sqrt((u-53)**2 + (v-44)**2))
			if D[u,v] <= Do:
				H[u,v] = 0

			D[u,v] = np.abs(np.sqrt((u-77)**2 + (v-87)**2))
			if D[u,v] <= Do:
				H[u,v] = 0
				
	return H 

def notch3(image):
	x = [80, 176, 150, 186]
	y = [128, 128, 178, 178]
	Do = 10
	H = np.ones(image.shape)
	for u in range(fshift.shape[0]):
		for v in range(fshift.shape[1]):

			D[u,v] = np.abs(np.sqrt((u-128)**2 + (v-80)**2))
			if D[u,v] <= Do:
				H[u,v] = 0

			D[u,v] = np.abs(np.sqrt((u-128)**2 + (v-176)**2))
			if D[u,v] <= Do:
				H[u,v] = 0

			D[u,v] = np.abs(np.sqrt((u-178)**2 + (v-150)**2))
			if D[u,v] <= Do:
				H[u,v] = 0

			D[u,v] = np.abs(np.sqrt((u-178)**2 + (v-186)**2))
			if D[u,v] <= Do:
				H[u,v] = 0
				
	return H 
if method == 'lowpass':
	H = lowpass(fshift)
if method == 'butterworth':
	H = butterworth(fshift)
if method == 'gaussian':
	H = gaussian(fshift)
if method == 'highpass_butterworth':
	H = highpass_butterworth(fshift)
if method == 'highpass_gaussian':
	H = highpass_gaussian(fshift)
if method == 'highpass':
	H = highpass(fshift)
if method == 'notch1':
	H = notch1(fshift)
if method == 'notch2':
	H = notch2(fshift)
if method == 'notch3':
	H = notch3(fshift)
if method == 'laplacian':
	H = laplacian(fshift)


filtered = np.multiply(fshift, H)
magnitude_filtered = np.abs(filtered) 
f_ishift = np.fft.ifftshift(filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

magnitude_spectrum = 20*np.log(np.abs(fshift))
magnitude_imgback = 20*np.log(np.abs(np.fft.fftshift(np.fft.fft2(img_back))))

#pdb.set_trace()
plt.subplot(141),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(magnitude_imgback, cmap = 'gray')
plt.title('Magnitude filtered'), plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(img_back, cmap = 'gray')
plt.title('%s'%method), plt.xticks([]), plt.yticks([])
plt.show()
#cv2.imshow('filtered', img_back.astype("uint8"))
#cv2.waitKey(0)