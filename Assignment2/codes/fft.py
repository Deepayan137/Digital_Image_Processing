import os
import numpy as np 
import argparse
import cv2
import pdb
from matplotlib import pyplot as plt
#fx = np.array([11, 23, 5, 15, 28, 17, 45, 41])
ap = argparse.ArgumentParser()
ap.add_argument("-I1", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def fft(fx):
	
	fx = np.asarray(fx, dtype=float)
	N = fx.shape[0]
	n= np.arange(N)
	n_even = n[::2]
	n_odd = n[1::2]
	if N>16:
		X_e = np.array([fx[n] for n in n_even])
		X_o = np.array([fx[n] for n in n_odd])
		X_e = fft(X_e)
		X_o = fft(X_o)
		factor = np.exp(-2j * np.pi * np.arange(N) / N)
		#pdb.set_trace()
		return np.concatenate([X_e + factor[:(N//2)] * X_o,
           		X_e + factor[(N//2):] * X_o])
	else:
		return DFT_slow(fx)


def fft2(image):
	
	#pdb.set_trace()
	image = cv2.resize(image,(256,256))
	h, w = image.shape
	fft_cols = np.zeros(image.shape)
	fft_rows = np.zeros(image.shape)
	for col in range(w):

		fft_cols[:, col] = fft(image[:,col])

	for row in range(fft_cols.shape[0]):
		fft_rows[row,:] = fft(image[row,:])

	return fft_rows

if __name__ == '__main__':
	image = cv2.imread(args["image"],0)
	fft_img = fft2(image)
	fshift = np.fft.fftshift(fft_img)
	magnitude_spectrum = 20*np.log(np.abs(fshift))
	plt.subplot(121),plt.imshow(image, cmap = 'gray')
	plt.title('Input Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
	plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
	plt.show()
	cv2.imshow('fft', fshift)
	cv2.waitKey(0)
	cv2.imwrite('fft.jpg', fshift)