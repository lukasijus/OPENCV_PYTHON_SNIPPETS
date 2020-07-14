import numpy as np
import cv2 as cv
im = cv.imread('Lenna.png')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

for i in range(100,105):
    ret, thresh = cv.threshold(imgray, i, 255, 0)
    cv.imshow('Tresh ' + str(i), thresh)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# To draw all the contours in an image:
ALL_CONTOURS = cv.drawContours(im, contours, -1, (0,255,0), 3)
# To draw an individual contour, say 4th contour:
UNI_COUNTOURS = cv.drawContours(im, contours, 3, (0,255,0), 3)
# But most of the time, below method will be useful:
cnt = contours[4]
USEFUL_COUNTOURS = cv.drawContours(im, [cnt], 0, (0,255,0), 3)

STACK = np.concatenate((thresh, imgray), axis=1)

cv.imshow('STACK', STACK)
cv.waitKey()


