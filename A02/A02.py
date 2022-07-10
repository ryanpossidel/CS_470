'''
Written by Ryan Possidel
Language: Python
Compiler: Anaconda
Description: manual slicing
Date: 10/7/21
'''

import cv2
import numpy as np
import sys
from pathlib import Path


def applyFilter(image,kernal):
    # changes variables to float64
    image = image.astype("float64")
    kernal = kernal.astype("float64")

    # flips the kernal for convolution
    kernal = cv2.flip(kernal, -1)

    # gets the filter height and width and half of them
    (filterHeight, filterWidth) = kernal.shape
    halfFH = int(filterHeight/2)
    halfFW = int(filterWidth/2)

    # Padded the image with a border of zeros
    borderImage = cv2.copyMakeBorder(image, halfFH, halfFH, halfFW, halfFW, borderType=cv2.BORDER_CONSTANT, value=0)

    # copies image to output
    output = np.copy(image)

    # gets the rows and columns of the padded image
    rows = image.shape[0]
    cols = image.shape[1]
    i = 0
    j = 0

    # covolution loop
    for i in range(rows):
        for j in range(cols):
            subImage = borderImage[i:(i + filterHeight), j:(j + filterWidth)]
            value = subImage * kernal
            output[i,j] = np.sum(value)

    return output

def main():
    # Checks if there are enough arguments
    if len(sys.argv) < 7:
        exit("ERROR: TOO FEW ARGUMENTS")

    # assigns arguments to the variables
    imgDir = sys.argv[1]
    outputDir = sys.argv[2]
    numRows = int(sys.argv[3])
    numCols = int(sys.argv[4])
    alphaVal = float(sys.argv[5])
    betaVal =  float(sys.argv[6])
    kernal = np.array(sys.argv[7:])

    # checks to see if the entered enough after the first 7
    if len(sys.argv) < 7 + (numRows*numCols):
        exit("ERROR: TOO FEW ARGUMENTS")

    # gets kernal in correct rows and columns
    kernal = kernal.reshape((numRows,numCols))

    # checks if there is an image
    if Path(imgDir).exists():
        # loads the image and changes it to grayscale
        image = cv2.imread(imgDir, cv2.IMREAD_GRAYSCALE)
    else:
        # prints error message and exits if there is not a valid filename
        print("ERROR: NO IMAGE")
        exit()

    # calls function and assigns what it returns to output
    output = applyFilter(image, kernal)

    output = cv2.convertScaleAbs(output, alpha=alphaVal, beta=betaVal)

    # writes the output to the output directory
    cv2.imwrite(outputDir, output)


if __name__ == "__main__":
    main()