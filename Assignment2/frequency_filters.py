import cv2
from matplotlib import pyplot as plt
import argparse
import pdb
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("-I1", "--image1", required=True, help="Path to the image")
args = vars(ap.parse_args())

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
def butterworth(image, n):
	for u in range(fshift.shape[0]):
		for v in range(fshift.shape[1]):
			D[u,v] = np.abs(np.sqrt((u-center[0])**2 + (v-center[1])**2))
			H[u,v] = 1/(1+ (D[u,v]/Do)**2*n)
	return H

H = butterworth(fshift,1)
filtered = np.multiply(fshift, H)
magnitude_filtered = np.abs(filtered) 
f_ishift = np.fft.ifftshift(filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
magnitude_spectrum = 20*np.log(np.abs(fshift))


#pdb.set_trace()
plt.subplot(141),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(magnitude_filtered, cmap = 'gray')
plt.title('Magnitude filtered'), plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(img_back, cmap = 'gray')
plt.title('low pass'), plt.xticks([]), plt.yticks([])
#plt.show()
cv2.imshow('orignal', img.astype("uint8"))
cv2.imshow('filtered', img_back.astype("uint8"))
cv2.waitKey(0)