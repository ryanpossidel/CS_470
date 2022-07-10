import numpy as np
import cv2
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import sys

def find_WBC(image):

    #kmeans
    pixels = np.reshape(image, (-1, 3)).astype("float32")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.90)
    retval, labels, centers = cv2.kmeans(pixels, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    results = centers[labels.flatten()]
    results = results.reshape(image.shape)

    #finding the center closest to red
    target = np.array([0,0,255])
    distance = np.empty(4)
    for i in range(len(centers)):
        current = centers[i]
        diff = target - current
        diff *= diff
        distance[i] = np.sqrt(np.sum(diff))


    for i in range(len(distance)):
        if distance[i] == min(distance):
            num = i
            break

    #reshapes labels
    new_image=labels.reshape(480,640)

    #gets x and y coords of the center cloest to red
    ycoords, xcoords = np.where(new_image == num)

    all_bb =[]
    #compute min max of x and y
    bb =[min(ycoords), min(xcoords), max(ycoords), max(ycoords)]
    all_bb.append(bb)

    return all_bb
