# MIT LICENSE
#
# Copyright 2020 Michael J. Reale
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pandas as pd
import sys
import tensorflow.keras.utils as utils
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC

def prepareTrainDataframe(frame):
    # Replace any data:
    # -- "cat" --> 0
    # -- "dog" --> 1
    frame["Filename"] = frame["Filename"].replace(to_replace=r'^cat.*', value='0', regex=True)
    frame["Filename"] = frame["Filename"].replace(to_replace=r'^dog.*', value='1', regex=True)
    frame["Filename"] = frame["Filename"].astype(float)
    
    # Get ground truth data
    dataY = frame["Filename"].values
    
    # Drop the ground truth column and get the actual values
    data = frame.drop(["Filename"], axis=1)
    dataX = data.values

    return dataX, dataY

def getMetrics(ground, pred, classCnt):
    acc = accuracy_score(y_true=ground, y_pred=pred)
    f1 = f1_score(y_true=ground, y_pred=pred, average='macro')
    one_hot_ground = utils.to_categorical(ground, num_classes=classCnt)
    one_hot_pred = utils.to_categorical(pred, num_classes=classCnt)
    auc = roc_auc_score(y_true=one_hot_ground, y_score=one_hot_pred, multi_class='ovr')
    return acc, f1, auc

def main():
    # Do we have enough arguments?
    if len(sys.argv) < 2:
        print("ERROR: Need <feature csv>!")
        exit(1)

    # Read in data
    print("Loading and preparing data...")
    allDataFilename = sys.argv[1]    
    allDataFrame = pd.read_csv(allDataFilename)    
    dataX, dataY = prepareTrainDataframe(allDataFrame)

    # Split into training and testing sets
    trainX, testX, trainY, testY = train_test_split(dataX, dataY, train_size=0.70, random_state=42, stratify=dataY)
    print("Training/testing split...")
    print("Training samples:", trainX.shape[0])
    print("Testing samples:", testX.shape[0])
        
    ##########################################################
    # TODO: Pick classifier!!!
    #classifier = "Choose wisely..." 
    #classifier = SVC(gamma=2, C=1)
    #classifier = SVC()
    classifier = KNeighborsClassifier(7)
    #classifier = NearestCentroid()
    ##########################################################

    # Train on data
    print("Starting training...")    
    classifier.fit(trainX, trainY)    
    print("Training complete.")

    # Get predictions
    print("Getting predictions...")
    predTrain = classifier.predict(trainX)
    predTest = classifier.predict(testX)

    # Get metrics for training
    accTrain, f1Train, aucTrain = getMetrics(predTrain, trainY, 2)        
    accTest, f1Test, aucTest = getMetrics(predTest, testY, 2)
    
    print("*****************************")
    print("TRAINING METRICS:")
    print("ACCURACY:", accTrain)
    print("F1 SCORE:", f1Train)
    print("AUC:", aucTrain)
    print("*****************************")
    print("TESTING METRICS:")
    print("ACCURACY:", accTest)
    print("F1 SCORE:", f1Test)
    print("AUC:", aucTest)
    print("*****************************")


if __name__ == "__main__":
    main()
