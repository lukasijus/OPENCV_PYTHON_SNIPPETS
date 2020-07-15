import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    scale_percent = 40  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized_frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
    resized_mask = cv.resize(mask, dim, interpolation=cv.INTER_AREA)
    resized_res = cv.resize(res, dim, interpolation=cv.INTER_AREA)
    cv.imshow('frame',resized_frame)
    cv.imshow('mask',resized_mask)
    cv.imshow('res',resized_res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()
