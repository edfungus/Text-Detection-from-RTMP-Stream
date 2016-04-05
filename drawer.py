from __future__ import division

import numpy as np
import cv2
import time

##
#   Drangonfly - Takes regions and draws them
#   Draws regions on given image
#   Edmund
##


# Simply draws given rectangles
#
# input: image to draw on, array of found text, color in which o box, thickness of box
# output: image with rectangles
def drawRectangles(image,rectArray,color,thickness):
    output = np.copy(image)

    for rectangle in rectArray:
        x = rectangle[0]
        y = rectangle[1]
        w = rectangle[2]
        h = rectangle[3]
        cv2.rectangle(output, (x,y), (x+w,y+h), color, thickness);

    return output

# draws rectangles at different colors based on additional input
#
# input: image to draw on, array of rectangles, array of colors where values 0 to 1 in [r,g,b] formation
# output: final image
def drawRectanglesWithColors(image,rectArray,colorArray,thickness):
    output = np.copy(image)
    blue = colorArray[:,:1]*255
    green = colorArray[:,1:2]*255
    red = colorArray[:,2:3]*255

    # draw rectangles with color corresponding to confidences
    for index, rectangle in enumerate(rectArray):
        x = rectangle[0]
        y = rectangle[1]
        w = rectangle[2]
        h = rectangle[3]
        color = (int(blue[index]),int(green[index]),int(red[index]))
        cv2.rectangle(output, (x,y), (x+w,y+h), color, thickness);

    return output
