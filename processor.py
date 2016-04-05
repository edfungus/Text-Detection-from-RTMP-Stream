from __future__ import division

import numpy as np
import cv2

##
#   Drangonfly - Process given regions with different functions
#   Different functions that takes regions and tries to generate useful combined values
#   Edmund
##

# Finds rectangle's similarity to other rectangles. The closer the box is to yellow, the more similar it is.
# The assumption is that text is more likely to be horizontally and vertically aligned to some other text,
# therefore, the more alignment, the more likely it is text
#
# input: array of found text, threshold of closeness of alignment, threshold only for display to highlight most unaligned objects
# output: image with rectangels, array of confidence in [[x,y],[x,y]...]
def alignmentTextConfidence(rectArray,sensitivityThreshold=100):
    confidence = np.array([[0,0]])

    # get confidence by searching array for similar horizontal and/or vertical boxes found by initial x,y coords
    for rectangle in rectArray:
        similarX = np.absolute(rectArray - rectangle)[:,:1]
        similarY = np.absolute(rectArray - rectangle)[:,1:2]
        countX = (similarX <= sensitivityThreshold).sum()
        countY = (similarY <= sensitivityThreshold).sum()
        confidence = np.vstack([confidence,[countX,countY]])
    confidence = confidence[1:]

    if confidence.size:
        # convert the counts of similar x,y coords to confidence between 0 and 1
        x = (confidence[:,:1]-confidence[:,:1].min())/confidence[:,:1].max()
        y = (confidence[:,1:2]-confidence[:,1:2].min())/confidence[:,1:2].max()
        confidence = np.append(x,y,axis=1)

    return confidence

# Find rectangle's closeness to other rectangles based on their centers
# ISSUE: by using the centers, we loose the "connecting" property of text
# REBUTAL^: the alighment confidence should take care of it
#
# input: array of found text, closeness threshold, threshold only for display to highlight most unclose objects
# output: array of confidence in [[confidence],[confidence]...] from 0 to 1
def closenessTextConfidence(rectArray,distanceThreshold=250):
    confidence = np.array([0])

    # get x,y of centers
    xy = rectArray[:,:2]
    wh = rectArray[:,2:4]
    centers = xy + wh/2

    # get the distance and sum up the occurances that there are other text blocks within a certian threshold
    for center in centers:
        xyDistance = np.absolute(center - centers)
        distance = np.power(np.power(xyDistance[:,:1],2) + np.power(xyDistance[:,1:2],2),.5)
        count = (distance <= distanceThreshold).sum()
        confidence = np.vstack([confidence,[count]])
    confidence = confidence[1:]

    if confidence.size:
        confidence = (confidence-confidence.min())/confidence.max()

    return confidence

# Use the w,h to give higher confidence to those with a rectangular shape
#
# input: array of found text, threshold to floor to 0 for any ratio below, multiplier that makes sharper cutoffs
# output: array of confidence in [[confidence],[confidence]...] from 0 to 1
def simpleWHRatio(rectArray,threshold=1,suppressor=1):
    confidence = rectArray[:,2:3] / rectArray[:,3:4]   # w/h
    # confidence[confidence <= threshold] = 1
    # confidence = np.log(suppressor*confidence)
    confidence = (confidence-confidence.min())/confidence.max()

    return confidence

#
def combineDiff(rectArray,image,noise,dilate,erode):
    mask = np.zeros((image.shape[0],image.shape[1]))
    combinedDiff = np.array([[0,0,0,0]])

    # set all changed regions to 1
    for rect in rectArray:
        x = rect[0]
        y = rect[1]
        w = rect[2]
        h = rect[3]
        mask[y:y+h,x:x+w] = 255

    bw = np.array(mask, np.uint8)

    # remove small ones
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (noise,noise))
    bw = cv2.erode(bw,kernel,iterations = 1)

    # combine
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate,dilate))
    bw = cv2.morphologyEx(bw, cv2.MORPH_DILATE, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode,erode))
    bw = cv2.morphologyEx(bw, cv2.MORPH_ERODE, kernel)


    im, contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is not None:
        idx = 0
        hierarchy = hierarchy[0]

        while idx >= 0:
            x,y,w,h = cv2.boundingRect(contours[idx])
            combinedDiff = np.vstack([combinedDiff,[x,y,w,h]])
            idx = hierarchy[idx][0]

    return combinedDiff[1:]

# Returns images of the parts that have changed
def getDiffImages(rectArray,completeFrame,wMin,hMin):
    images = []
    for rect in rectArray:
        x = rect[0]
        y = rect[1]
        w = rect[2]
        h = rect[3]
        if w > wMin and h > hMin:
            obj = {}
            obj["location"] = rect
            obj["image"] = completeFrame[y:y+h,x:x+w]
            images.append(obj)

    return images

# Return the coords based on original image of changed parts
def convertLocalCoordsToImageCoords(rectArray, diffPartObj):
    rectArray = np.copy(rectArray)
    xAdj = diffPartObj["location"][0]
    yAdj = diffPartObj["location"][1]
    for rect in rectArray:
        rect[0] = rect[0] + xAdj
        rect[1] = rect[1] + yAdj

    return rectArray

def cleanDiff(oldDiff, changeDiff,image):
    mask = np.zeros((image.shape[0],image.shape[1]))
    cleanedDiff = np.array([[0,0,0,0]])

    # set all changed regions to 1
    for change in changeDiff:
        x = change[0]
        y = change[1]
        w = change[2]
        h = change[3]
        mask[y:y+h,x:x+w] = 1

    # check if oldDiff overlaps with
    for old in oldDiff:
        x = old[0]
        y = old[1]
        w = old[2]
        h = old[3]
        if np.sum(mask[y:y+h,x:x+w]) == 0:
            cleanedDiff = np.vstack([cleanedDiff,[old]])

    return cleanedDiff[1:]
