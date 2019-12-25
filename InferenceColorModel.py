import numpy as np
import os
import cv2
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import heapq

from tensorflow.keras.models import load_model



dataPrePath = "e:\\projects\\MB\\ColorNitzan\\ATR\\"
modelFile = 'color_classification_smaller_ALL_DATA.h5'
testImagesDir = os.path.join(dataPrePath, r'Database_with_MB\testset')
testImage = os.path.join(testImagesDir,"green\private_2166.png")
image_resolution=(128,128)
hotEncode = {'white': 0, 'black': 1, 'gray': 2, 'red': 3, 'green': 4, 'blue': 5,
                          'yellow': 6, 'cyan': 5}
hotEncodeReverse = {0: 'white', 1: 'black', 2: 'gray', 3: 'red', 4: 'green', 5: 'blue',
                                 6: 'yellow'}

#load model from file
cModel = load_model(modelFile)

#print info about model
cModel.summary()

#load one image from files
img = cv2.resize(cv2.imread(testImage),image_resolution)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

imgf = img.astype(float)/255.0
prediction = cModel.predict(imgf.reshape([1,128,128,3]), verbose=0)
print("Predicted color:" + hotEncodeReverse[prediction.argmax()] + ", score =  " + str(prediction.max()) )
print("Second option:" + hotEncodeReverse[prediction.argsort()[0,-2]] + ", score = " + str(prediction[0,prediction.argsort()[0,-2]]))

####full load dataset#####
folderList = sorted(os.listdir(testImagesDir))
for idx, folder in enumerate(folderList):
    folderPath = testImagesDir + r'/' + folder
    fileList = os.listdir(folderPath)
    for idx2, file in enumerate(fileList):
        filePath = os.path.abspath(os.path.join(folderPath, file))
        img = cv2.resize(cv2.imread(filePath), image_resolution)
        imgf = img.astype(float) / 255.0
        prediction = cModel.predict(imgf.reshape([1, 128, 128, 3]), verbose=0)
        print("***" + folder + "****")
        print("Predicted color:" + hotEncodeReverse[prediction.argmax()] + ", score =  " + str(prediction.max()))
        print("Second option:" + hotEncodeReverse[prediction.argsort()[0, -2]] + ", score = " + str(prediction[0, prediction.argsort()[0, -2]]))
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



