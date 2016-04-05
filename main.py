from __future__ import division

import cv2
import numpy as np

import rtmp
import analyzer
import drawer
import processor

##
#   Drangonfly - Main video analyzer
#   Takes a video and analyze it for features
#   Edmund
##

capture = rtmp.captureVideo("rtmp://192.168.1.139:1935/live/edmund live=1 buffer=10")
oldFrame = rtmp.getFrame(capture)#[92:,:]
currentText = np.array([[0,0,0,0]])
frameSkipCount = 0
displaySkipCount = 0

def combineMultiple(rectArray):
    confidAlignment = processor.alignmentTextConfidence(cannyTextRect,100)
    confidCloseness = processor.closenessTextConfidence(cannyTextRect,250)
    confidRatio = processor.simpleWHRatio(cannyTextRect,1.2,1)

    confidCombined = confidAlignment[:,:1]/(35/100) + confidAlignment[:,1:2]/(35/100) + confidCloseness/(20/100) + confidRatio/(10/100)
    # confidCombined = confidAlignment[:,:1]/4 + confidAlignment[:,1:2]/4 + confidCloseness/4 + confidRatio/4
    confidCombined = (confidCombined - confidCombined.min())/confidCombined.max()

    return confidCombined


while(True):
    #** get frame
    currentFrame = rtmp.getFrame(capture)#[92:,:]   #[68:,:]   #[92:,:]

    if frameSkipCount == 0:
        frameSkipCount = 0
    else:
        frameSkipCount += 1
        continue

    grayFrame = cv2.cvtColor(cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    black = np.zeros(currentFrame.shape)
    displayFrame = np.copy(currentFrame)

    #** analyze frames
    # text = analyzer.gradientText(currentFrame,5,10,100,.4)
    # cannyTextRect = analyzer.cannyText(currentFrame,5,10,100,1000,.4)

    #** create output frame
    # displayFrame = drawer.drawRectangles(displayFrame,cannyTextRect,(255,0,255),2)
    # output2 = drawer.drawRectangles(grayFrame,cannyText,(0,255,255),3)

    # Alignment testing
    # confidAlignment = processor.alignmentTextConfidence(cannyTextRect,100)
    # zeroPad = np.zeros((confidAlignment.shape[0],1))
    # confidAlignment = np.append(zeroPad,confidAlignment,axis=1)
    # confidOutput = drawer.drawRectanglesWithColors(grayFrame,cannyTextRect,confidAlignment,2)

    # Alignment testing (COMBINED)
    # confidAlignment = processor.alignmentTextConfidence(cannyTextRect,100)
    # confidAlignmentCombined = confidAlignment[:,:1]/2 + confidAlignment[:,1:2]/2
    # zeroPad = np.zeros((confidAlignment.shape[0],2))
    # confidAlignmentCombined = np.append(zeroPad,confidAlignmentCombined,axis=1)
    # confidOutput = drawer.drawRectanglesWithColors(grayFrame,cannyTextRect,confidAlignmentCombined,2)


    # Closeness testing
    # confidCloseness = processor.closenessTextConfidence(cannyTextRect,250)
    # zeroPad = np.zeros((confidCloseness.shape[0],2))
    # confidCloseness = np.append(zeroPad,confidCloseness,axis=1)
    # confidOutput = drawer.drawRectanglesWithColors(grayFrame,cannyTextRect,confidCloseness,2)

    # Combine both
    # confidCombine = np.append(confidCloseness,confidAlignment,axis=1)
    # confidOutput = drawer.drawRectanglesWithColors(currentFrame,cannyTextRect,confidCombine,2)

    # w,h ratio
    # confidRatio = processor.simpleWHRatio(cannyTextRect,1.2,1)
    # zeroPad = np.zeros((confidRatio.shape[0],2))
    # confidRatio = np.append(zeroPad,confidRatio,axis=1)
    # confidOutput = drawer.drawRectanglesWithColors(grayFrame,cannyTextRect,confidRatio,2)

    # combine multiplier
    # confid = combineMultiple(cannyTextRect)
    # zeroPad = np.zeros((confid.shape[0],2))
    # confid = np.append(zeroPad,confid,axis=1)
    # confidOutput = drawer.drawRectanglesWithColors(grayFrame,cannyTextRect,confid,2)
    #
    # newDiff = analyzer.simpleDifference(oldFrame,currentFrame)
    # if newDiff.size:
    #     if currentText.size:
    #         combinedNewDiff = processor.combineDiff(newDiff,currentFrame,5,150,100)
    #         currentText = processor.cleanDiff(currentText,combinedNewDiff,currentFrame)  # remove old box in changed areas
    #         imagesForProcess = processor.getDiffImages(combinedNewDiff,currentFrame,10,10)   # get image of each newly changed area
    #
    #         # process newly changed area
    #         for imageObj in imagesForProcess:
    #             cannyTextRectLocale = analyzer.cannyText(imageObj["image"],5,10,100,1000,.4)    #find text
    #             imageCoordRectArray = processor.convertLocalCoordsToImageCoords(cannyTextRectLocale,imageObj) #conver local coords to image coords
    #             currentText = np.vstack([currentText,imageCoordRectArray])
    #
    #         displayFrame = drawer.drawRectangles(displayFrame,combinedNewDiff,(255,0,255),2)
    #
    #     else:
    #         currentText = newDiff
    #
    # displayFrame = drawer.drawRectangles(displayFrame,currentText,(0,0,255),2)
    #
    # if displaySkipCount == 19:
    #     displaySkipCount = 0
    #     cv2.imshow("video", displayFrame)
    # else:
    #     displaySkipCount += 1
    #     continue

    cv2.imshow("video", displayFrame)
    oldFrame = currentFrame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
