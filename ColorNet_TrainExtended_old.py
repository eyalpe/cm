import numpy as np
import os
import cv2
import shutil
import platform
import k2tf

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, Lambda
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K

from jointDataset import chenColorDataset, dataSetHistogram
import datetime

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import freezeUtils

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

if(platform.system()=="Windows"):
    dataPrePath = r"e:\\projects\\MB\\ColorNitzan\\ATR\\"
    outputPath = r"e:\\projects\\MB\\ColorNitzan\\TFexample\\outColorNetOutputs_30_01_20\\"
else:
    if(os.getlogin()=='borisef'):
        dataPrePath = "/media/borisef/d1e28558-1377-4bbb-9e48-c8900feaf59d/ColorNitzan/ATR/"
        outputPath = "/home/borisef/projects/cm/Output/outColorNetOutputs_08_03_20_full"



if os.path.exists(outputPath):
    shutil.rmtree(outputPath)
os.mkdir(outputPath)

stat_save_dir = os.path.join(outputPath,"stat")
simple_save_dir = os.path.join(outputPath,"simpleSave")
frozen_dir = os.path.join(outputPath,"frozen")
model_n_ckpt_dir = os.path.join(outputPath,"model")
ckpt_dir = os.path.join(model_n_ckpt_dir,"checkpoint")
h5_dir = os.path.join(outputPath,"h5")
k2tf_dir = os.path.join(outputPath,"k2tf")

os.mkdir(model_n_ckpt_dir)
os.mkdir(stat_save_dir)
os.mkdir(ckpt_dir)
os.mkdir(frozen_dir)
os.mkdir(h5_dir)
os.mkdir(k2tf_dir)



trainSet = chenColorDataset(os.path.join(dataPrePath, r'Database_clean_unified_augmented4'), gamma_correction=False)
testSet = chenColorDataset(os.path.join(dataPrePath, r'Database_with_MB/testset'), gamma_correction=False)
dataSetHistogram(trainSet.allData['labels'], trainSet.hotEncodeReverse, os.path.join(stat_save_dir,"hist.png"))

#Model Architecture
model = Sequential()
model.add(Convolution2D(16, 3, 3, activation='relu', input_shape=(128, 128, 3)))
model.add(BatchNormalization())
model.add(Convolution2D(16, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
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
model.add(Lambda(lambda x: x, name='colors_prob'))

model.summary()
#categorical_crossentropy
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


saver = tf.train.Saver()

with open(os.path.join(model_n_ckpt_dir,'model.pb'), 'wb') as f:
    f.write(tf.keras.backend.get_session().graph_def.SerializeToString())


model.fit(trainSet.allData['images'], trainSet.allData['labels'], batch_size=256, nb_epoch=5, verbose=1)
t0 = now()
test_loss, test_acc = model.evaluate(testSet.allData['images'], testSet.allData['labels'], verbose=0)
dt = now()-t0
print("Score: {}, evaluation time: {}, time_per_image: {}".format(test_acc, dt, dt/len(testSet.allData['labels'])))

#save model
model.save(os.path.join(h5_dir,'color_classification_smaller_ALL_DATA.h5'))



tf.saved_model.simple_save(tf.keras.backend.get_session(),
                           simple_save_dir,
                           inputs={"input": model.inputs[0]},
                           outputs={"output": model.outputs[0]})



# Saver
saver.save(tf.keras.backend.get_session(), os.path.join(ckpt_dir,"train.ckpt"))

try:
    freeze_graph.freeze_graph(None,
                              None,
                              None,
                              None,
                              model.outputs[0].op.name,
                              None,
                              None,
                              os.path.join(frozen_dir, "frozen_cmodel.pb"),
                              False,
                              "",
                              input_saved_model_dir=simple_save_dir)

except:
    print("freeze_graph.freeze_graph FAILED")

try:
    #save model
    model.save(os.path.join(frozen_dir, 'color_classification_smaller_ALL_DATA'), save_format='tf')
except:
    try:
        model.save(os.path.join(frozen_dir, 'color_classification_smaller_ALL_DATA'))
    except:
        print("model.save(...,  save_format='tf')  FAILED")

print(model.outputs)
print(model.inputs)

try:
    frozen_graph1 = freezeUtils.freeze_session(K.get_session(),
                                   output_names=[out.op.name for out in model.outputs])
except:
    print("frozen_graph1 = freezeUtils.freeze_session...  FAILED")

# Save to ./model/tf_model.pb
try:
    tf.train.write_graph(frozen_graph1, "model", "tf_model.pb", as_text=False)
except:
    print("tf.train.write_graph(frozen_graph1..  FAILED")


try:
    args_model = os.path.join(h5_dir,'color_classification_smaller_ALL_DATA.h5')
    args_num_out = 1
    args_outdir = k2tf_dir
    args_prefix = "k2tfout"
    args_name = "output_graph.pb"

    k2tf.convertGraph(args_model, args_outdir, args_num_out, args_prefix, args_name)

except:
    print("2tf.convertGraph  FAILED")













#import pdb; pdb.set_trace()
M = confusion_matrix(model, testSet.allData)
print(M)
show_conf_matr(M, os.path.join(stat_save_dir,"conf.png"))
for idx, image in enumerate(testSet.allData['images']):
    im_rs = cv2.resize(image, (360, 360))
    prediction = model.predict_classes(testSet.allData['images'][idx].reshape([1,128,128,3]), verbose=0)
    print("{}/{}:   {}".format(idx+1, len(testSet.allData['images']), testSet._return_label(testSet.allData['labels'][idx])))
    cv2.imshow(testSet.hotEncodeReverse[prediction[0]], im_rs)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()