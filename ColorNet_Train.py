import numpy as np
import os
import cv2
import shutil

#from keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from jointDataset import chenColorDataset, dataSetHistogram
import datetime

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def show_conf_matr(M, outf):
    df_cm = pd.DataFrame(M, range(len(M)),
                         range(len(M)))
    # plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size

    plt.savefig(outf)
    plt.close()
    #plt.show()



now = datetime.datetime.now


def confusion_matrix(model, testSet):
    hist = np.sum(testSet['labels'], axis=0)
    size_matrix = np.repeat(hist, repeats=len(hist)).reshape(len(hist), len(hist))
    conf = np.zeros(size_matrix.shape)
    for idx, image in enumerate(testSet['images']):
        prediction = model.predict_classes(testSet['images'][idx].reshape([1, 128, 128, 3]), verbose=0)[0]
        label = np.where((testSet['labels'][idx] == 1))[0][0]
        conf[label][prediction] += 1
    conf /= size_matrix
    return conf


dataPrePath = r"e:\\projects\\MB\\ColorNitzan\\ATR\\"
outputPath = "outColorNet"

if(os.path.exists(outputPath)):
    shutil.rmtree(outputPath)
os.mkdir(outputPath)

trainSet = chenColorDataset(os.path.join(dataPrePath, r'Database_clean_unified_augmented4mini'), gamma_correction=False)
testSet = chenColorDataset(os.path.join(dataPrePath, r'Database_with_MB\testset'), gamma_correction=False)
dataSetHistogram(trainSet.allData['labels'], trainSet.hotEncodeReverse, os.path.join(outputPath,"hist.png"))

#Model Architecture
model = Sequential()
model.add(Convolution2D(16,3,3, activation='relu', input_shape=(128,128,3)))
model.add(BatchNormalization())
model.add(Convolution2D(16,3,3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(BatchNormalization())
model.add(Dense(len(trainSet.hotEncodeReverse), activation='softmax'))

model.summary()

#categorical_crossentropy
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(trainSet.allData['images'], trainSet.allData['labels'], batch_size=256, nb_epoch=10, verbose=1)
t0 = now()
test_loss, test_acc  = model.evaluate(testSet.allData['images'], testSet.allData['labels'], verbose=0)
dt = now()-t0
print("Score: {}, evaluation time: {}, time_per_image: {}".format(test_acc, dt, dt/len(testSet.allData['labels'])))

#save model
model.save('color_classification_smaller_ALL_DATA.h5')

#import pdb; pdb.set_trace()
M = confusion_matrix(model, testSet.allData)
print(M)
show_conf_matr(M, os.path.join(outputPath,"conf.png"))
for idx, image in enumerate(testSet.allData['images']):
    im_rs = cv2.resize(image, (360, 360))
    prediction = model.predict_classes(testSet.allData['images'][idx].reshape([1,128,128,3]), verbose=0)
    print("{}/{}:   {}".format(idx+1, len(testSet.allData['images']), testSet._return_label(testSet.allData['labels'][idx])))
    cv2.imshow(testSet.hotEncodeReverse[prediction[0]], im_rs)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()