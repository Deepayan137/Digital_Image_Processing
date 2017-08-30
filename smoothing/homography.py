import cv2
import sys
import numpy as np 
import pdb

refPt = []
refPrime = []
ref=[]
def click(event, x, y, flags, param):
	global  cropping ,ix, iy, refPt 
	if event == cv2.EVENT_LBUTTONDOWN:
		ix, iy = x, y
		cv2.circle(image, (ix,iy),20, (0, 0, 255), 1)	
		cv2.imshow('image',image)
		refPt.append([x, y])
		ref.append([x, y])
	elif event == cv2.EVENT_LBUTTONUP:
		ix, iy = x, y
		refPrime.append([x -475, y])
		ref.append([x, y])
		cv2.circle(image, (ix,iy),20, (0, 0, 255), 1)
		cv2.imshow('image',image)
image_path = sys.argv[1]
image = cv2.imread(image_path)

image = cv2.resize(image, (960, 540))
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click)

while True:

	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("r"):
		image = clone.copy()

	elif key == ord("a"):
		
		cv2.line(image,(ref[0][0],ref[0][1]),
			(ref[1][0],ref[1][1]), [255, 0, 0], 1)
		point1 = np.append(np.array(ref[0]), np.array([1]))
		point2 = np.append(np.array((ref[1][0]-475,ref[1][0])),
			np.array([1]))
		product = np.outer(point1, 1./point2)
		ref=[]
		#print(product)

	elif key ==ord('d'):
		
		if len(refPt) == 3:
			
			for i in range(len(refPt)):
				refPt[i].extend([1])
				refPrime[i].extend([1])
			
			refPt = np.array(refPt)
			refPrime = np.array(refPrime)
			trans = np.dot(np.linalg.inv(refPt), refPrime)
			print(trans)
		else:
			print("3 pts needed")
	elif key == ord("q"):
		break
cv2.imwrite('stereo.jpg', image)
cv2.destroyAllWindows()




