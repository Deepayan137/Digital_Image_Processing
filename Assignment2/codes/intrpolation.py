import numpy as np 
import cv2
from matplotlib import pyplot as plt
import argparse
import pdb
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("-I1", "--image1", required=True, help="Path to the image")
ap.add_argument("-m", "--method", required=True, help="interpolating method")
ap.add_argument("-s", "--scale", type=int,default=2)
args = vars(ap.parse_args())



def n_neighbour(image, scale):
	print (scale)
	h, w = image.shape
	#pdb.set_trace()
	H, W = h*scale, w*scale
	
	M = np.zeros((H, W))
	print(image.shape)
	for y in range(H-1):
		for x in range(W-1):
			#pdb.set_trace()
			#print(int(y),int(x))
			M[y,x] = image[int(y//scale),int(x//scale)]

	return M
	
def bilinear(image, scale):
	h, w = image.shape
	H, W = h*scale, w*scale
	
	M = np.zeros((H, W))
	for y in range(H-3):
		for x in range(W-3):
			#pdb.set_trace()
			I12 = (1 - 1/scale)*image[int(y//scale),int(x//scale)] + (1/scale)*image[int(y//scale), int(x//scale)+1]
			I34 = (1 - 1/scale)*image[int(y//scale)+1, int(x//scale)] + (1/scale)*image[int(y//scale)+1, int(x//scale)+1]
			M[y,x] = (1 - 1/scale)*I12 +(1/scale)*I34
	return M

def cubic(I, x):
	a = -0.5*I[0] + 1.5*I[1] - 1.5*I[2]+ 0.5*I[3]
	b = I[0] - 2.5*I[1] + 2*I[2] - 0.5*I[3]
	c = -0.5*I[0] + 0.5*I[1]
	d = I[1]
	x = x-np.floor(x)
	value =   a*(x**3) + b*(x**2) + c*(x) + d
	return value

def bicubic(image, scale):
	h, w = image.shape
	H, W = int(h*scale),int(w*scale)
	M = np.zeros((H, W))
	I = np.zeros((4,4))
	for x in range(W-6):
		x_f = int(x/scale)
		for y in range(H-6):
			val =[]
			y_f = y//scale
			
			x1 = x_f
			x0 = max(0,x1 - 1)
			x2 = min(w,x_f+1)
			x3 = min(w,x2+1)

			
			y1 = y_f
			y0 = max(0,y1-1)
			y2 = min(h,y_f+1)
			y3 = min(h,y2+1)
			x_ = [x0, x1, x2, x3]
			y_= [y0, y1, y2, y3]
			for i in range(4):
				for j in range(4):
					I[i,j] = image[y_[j],x_[i]]
			
		
			val = [cubic(I[i],y_f)for i in range(4)]
			M[y,x] = cubic(val,x_f)
	return M	
def bilinear_ver02(image, scale):
	h, w = image.shape
	H, W = h*scale, w*scale
	print (H,W)
	M = np.zeros((H, W))
	for y in range(H-1):
		for x in range(W-1):
			W = -int(((x/scale)- np.floor(x/scale))-1)
			H = -int(((y/scale)- np.floor(y/scale))-1)
			#pdb.set_trace()
			I11 = image[int(np.floor(y/scale)),int(np.floor(x/scale))]
			I12 = image[int(np.ceil(y/scale)),int(np.floor(x/scale))]
			I21 = image[int(np.floor(y/scale)),int(np.ceil(x/scale))]
			I22 = image[int(np.ceil(y/scale)),int(np.ceil(x/scale))]
			M[y+1,x+1] = (1-W)*(1-H)*I22 + (W)*(1-H)*I21 + (1-W)*(H)*I12 + (W)*(H)*I11;
	return M
if __name__ == '__main__':
	image = cv2.imread(args['image1'], 0)
	scale = args['scale']
	method = args['method']
	if method == 'bicubic':
		M = bicubic(image, scale)
	if method == 'bilinear':
		M = bilinear(image, scale)
	if method == 'n_neighbour':
		M = n_neighbour(image, scale)
	plt.subplot(121),plt.imshow(image, cmap = 'gray')
	plt.title('Input Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(M, cmap = 'gray')
	plt.title('interpolated image'), plt.xticks([]), plt.yticks([])
	#plt.show()
	cv2.imshow('orignal',image)
	cv2.imshow('interpolated', M.astype("uint8"))
	cv2.waitKey(0)