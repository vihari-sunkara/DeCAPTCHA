import cv2 as cv
import numpy as np
from scipy import ndimage
img= cv.imread('AAA.png')
hsv= cv.cvtColor(img,cv.COLOR_BGR2HSV)
lower_p = np.array([0,0,150])
upper_p =np.array([255,255,255])
mask=cv.inRange(hsv, lower_p, upper_p) # background noise removed.
#res= cv.bitwise_and(hsv,img,mask=mask) NOT USEFUL
kernel = np.ones((5,5), np.uint8)  # erosion
img_erosion = cv.erode(mask, kernel, iterations=1) 
kernel_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]), np.array([1, 1, 1])]) # finding edges which may be useful in segmentation ,Can use some other technique also 
out_l = ndimage.convolve(img_erosion, kernel_laplace, mode='reflect')

cv.imshow('Erosion', img_erosion) 
cv.imshow('img',img)
cv.imshow('mask',mask)
#cv.imshow('res',res)
cv.imshow('edge', out_l) 
cv.waitKey(0)
cv.destroyAllWindows()
