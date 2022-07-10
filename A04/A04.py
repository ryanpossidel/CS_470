import cv2
import numpy as np

def getLBPImage(image):

    #makes border image
    borderImage = cv2.copyMakeBorder(image, 1, 1, 1, 1, borderType=cv2.BORDER_CONSTANT, value=0)

    #copies image to output
    output = np.copy(image)

    #LBP calculations
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):

            #gets subImage and gets neighbor pixels clockwise
            subImage = borderImage[row:(row + 3), col:(col + 3)]
            array = np.empty(8)
            array[0] = subImage[0][0]
            array[1] = subImage[0][1]
            array[2] = subImage[0][2]
            array[3] = subImage[1][2]
            array[4] = subImage[2][2]
            array[5] = subImage[2][1]
            array[6] = subImage[2][0]
            array[7] = subImage[1][0]

            #changes values to 1 or 0
            centerVal = subImage[1][1]
            i=0
            for i in range(len(array)):
                if centerVal < array[i]:
                    array[i] = 1
                else:
                    array[i] = 0

            #gets number of transitions
            transition = 0
            for j in range(len(array)-1):
                if array[j] != array[j+1]:
                    transition+=1

            #gets label number
            label = 0
            if transition > 2:
                label = 9
            else:
                for k in range(len(array)):
                    if array[k] == 1:
                        label+=1

            #assigns label to output pixel
            output[row][col] = label

    return output


def getOneRegionLBPFeatures(subImage):

    #creates a histogram of the given LBP image
    hist = cv2.calcHist([subImage], [0], None, [10], [0, 10])
    hist /= subImage.shape[0]*subImage.shape[1]
    hist = hist.reshape(10)
    return hist

def getLBPFeatures(featureImage, regionSideCnt):

    #gets the sub Regions width and height
    featureImgWidth, featureImgHeight = featureImage.shape
    subWidth = featureImgWidth//regionSideCnt
    subHeight = featureImgHeight//regionSideCnt
    allHists = []

    #loops through the sub regions
    for i in range(regionSideCnt):
        for j in range(regionSideCnt):

            #gets the coords of the subregions
            xStart = int(i * subWidth)
            xEnd = int((i + 1) * subWidth)
            yStart = int(j * subHeight)
            yEnd = int((j + 1) * subHeight)

            #assigns the coords to the sub region
            subRegion = featureImage[xStart:xEnd,yStart:yEnd]

            #calls getOneRegionLBPFeatures to make histogram of sub region
            hist=getOneRegionLBPFeatures(subRegion)
            allHists.append(hist)

    #changes the histogram to a np array and reshapes it
    allHists = np.array(allHists)
    allHists = np.reshape(allHists, (allHists.shape[0] * allHists.shape[1]))

    return allHists
