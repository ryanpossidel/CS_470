import cv2
import sys
import numpy as np
from pathlib import Path

def slice_image(image, lower_slice, upper_slice):
    #copies the image to output
    output = np.copy(image)

    #checks to see if the upper and lower slices are greater or less than the output
    output = np.where(output < lower_slice, 0 , output)
    output = np.where(output > upper_slice, 0 , output)

    return output

def main():

    #Checks if there are enough arguments
    if len(sys.argv) < 5:
        print("ERROR: TOO FEW ARGUMENTS")
        exit()

    #assigns the arguments to variables
    imgDir = sys.argv[1]
    lower_slice = int(sys.argv[2])
    upper_slice = int(sys.argv[3])
    outputDir = sys.argv[4]

    #checks if there is an image
    if Path(imgDir).exists():
        #loads the image and changes it to grayscale
        image = cv2.imread(imgDir, cv2.IMREAD_GRAYSCALE)
    else:
        #prints error message and exits if there is not a valid filename
        print("ERROR: NO IMAGE")
        exit()


    #calls the slicing function
    output = slice_image(image, lower_slice, upper_slice)

    #declares out_filename as a string
    out_filename = str()

    #makes the filename and appends it to the output directory
    out_filename += "OUT_" + Path(imgDir).stem + "_" + str(lower_slice) + "_" + str(upper_slice) + ".png"
    outputDir += "/" + out_filename

    #saves to outputdir with the filename
    cv2.imwrite(outputDir, output)

if __name__ == "__main__":
    main()


