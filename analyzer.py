from __future__ import division

import numpy as np
import cv2

##
#   Drangonfly - Analyzer of video frames
#   Assortment of functions that are used to analyzer the frames and returns feature locations or partial feature locations
#   Edmund
##

# Find the differences between two images
#
# input: image1, image2
# output: array for rectangle box [x,y,w,h] that specifies area of change
def simpleDifference(oldImage,newImage):
    # load images
    image1 = cv2.cvtColor(np.copy(oldImage), cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(np.copy(newImage), cv2.COLOR_BGR2GRAY)
    # image1 = np.copy(oldImage)
    # image2 = np.copy(newImage)

    # simple difference
    diff = cv2.absdiff(image2, image1); #diff of gray scale image may not work very well
    # diff = cv2.cvtColor(cv2.absdiff(image2, image1), cv2.COLOR_BGR2GRAY)
    ret, bw = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    # processing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closing = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations = 1)

    # getting regions
    mask = np.zeros(image1.shape, dtype=np.uint8)
    hierarchy = np.array([])
    im, contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = np.array([[0,0,0,0]])

    if hierarchy is not None:
        idx = 0
        hierarchy = hierarchy[0]

        while idx >= 0:
            x,y,w,h = cv2.boundingRect(contours[idx])
            regions = np.vstack([regions,[x,y,w,h]])
            idx = hierarchy[idx][0]


    return regions[1:]

# Find text by using the gradient morphology filter
#
# input: image to analyze, minimum height of text, minimum width of text, max height of text, percentage of area "text" it takes up in a rectangle
# output: array for rectangle box [x,y,w,h] that specifies area of text
def gradientText(image,minH,minW,maxH,threshold):
    # load image
    gray = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2GRAY)

    # morphological filter
    gradientKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, gradientKernel)

    # binarize
    ret, bw = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # connect horizontal regions
    connectedKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, connectedKernel)

    # getting text areas
    mask = np.zeros(bw.shape, dtype=np.uint8)
    im, contours, hierarchy = cv2.findContours(connected, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    regions = np.array([[0,0,0,0]])

    if hierarchy is not None:
        idx = 0
        hierarchy = hierarchy[0]

        while idx >= 0:
            x,y,w,h = cv2.boundingRect(contours[idx])

            if w > h and h > minH and w > minW and h < maxH:
                mask[y:y+h,x:x+w].fill(0.0)

                cv2.drawContours(mask, contours, idx, (255,255,255), cv2.FILLED)
                r = cv2.countNonZero(mask[y:y+h,x:x+w])/(w*h)

                # checking if the contoured region fills up a percentage of the rectangle
                if r > threshold:
                    regions = np.vstack([regions,[x,y,w,h]])

            idx = hierarchy[idx][0]

    return regions[1:]

# Find text by using the canny edge filter
#
# input: image to analyze, minimum height of text, minimum width of text, max height of text, max width of text, percentage of area "text" it takes up in a rectangle
# output: array for rectangle box [x,y,w,h] that specifies area of text
def cannyText(image,minH,minW,maxH,maxW,threshold):
    # load image
    gray = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2GRAY)

    # Canny filter
    bw = cv2.Canny(gray,200,255)

    # connect horizontal regions
    connectedKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, connectedKernel)

    # getting text areas
    mask = np.zeros(bw.shape, dtype=np.uint8)
    im, contours, hierarchy = cv2.findContours(connected, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    regions = np.array([[0,0,0,0]])

    if hierarchy is not None:
        idx = 0
        hierarchy = hierarchy[0]

        while idx >= 0:
            x,y,w,h = cv2.boundingRect(contours[idx])

            if w > h and h > minH and w > minW and h < maxH and w < maxW:
                mask[y:y+h,x:x+w].fill(0.0)

                cv2.drawContours(mask, contours, idx, (255,255,255), cv2.FILLED)
                r = cv2.countNonZero(mask[y:y+h,x:x+w])/(w*h)

                # checking if the contoured region fills up a percentage of the rectangle
                if r > threshold:
                    regions = np.vstack([regions,[x,y,w,h]])

            idx = hierarchy[idx][0]

    return regions[1:]
