import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
import copy

class jointDataSet:
    def __init__(self, list_of_datasets, trainSetPercentage=0.6, validationSetPercentage=0):
        L = len(list_of_datasets)
        self.listOfDataSets = list_of_datasets
        #check if categories are the same
        for dataset in list_of_datasets:
            if dataset.hotEncodeReverse != list_of_datasets[0].hotEncodeReverse or dataset.image_resolution != list_of_datasets[0].image_resolution:
                print("ERROR!!! dataset categories are different")
                import pdb; pdb.set_trace()
        self.hotEncodeReverse = list_of_datasets[0].hotEncodeReverse
        #append images and labels, then shuffle
        self.allData = {}
        self.allData['images'] = list_of_datasets[0].allData['images']
        self.allData['labels'] = list_of_datasets[0].allData['labels']
        print("concatenated 1/{}".format(L))
        for idx in range(1, L):
            self.allData['images'] = np.concatenate((self.allData['images'], list_of_datasets[idx].allData['images']), axis=0)
            self.allData['labels'] = np.concatenate((self.allData['labels'], list_of_datasets[idx].allData['labels']), axis=0)
            print("concatenated {}/{}".format(idx+1,L))
        ####Randomly select data for train, test and validation sets####
        # Shuffle Data
        randperm = np.random.permutation(np.arange(len(self.allData['images'])))
        trainIdx = int(trainSetPercentage * len(self.allData['images']))
        self.allData['images'] = np.array(self.allData['images'])[randperm]
        #self.allData['images'] = self.allData['images'].astype('float32')
        #self.allData['images'] /= 255.0
        self.allData['labels'] = np.array(self.allData['labels'])[randperm]
        #####create sets#####
        self.trainSet = {'images': self.allData['images'][:trainIdx], 'labels': self.allData['labels'][:trainIdx]}
        self._num_examples_train = len(self.trainSet['images'])
        if validationSetPercentage != 0:
            validationIdx = int((trainSetPercentage + validationSetPercentage) * len(self.allData['images']))
            self.validationSet = {'images': self.allData['images'][trainIdx:validationIdx],
                                  'labels': self.allData['labels'][trainIdx:validationIdx]}
            self.testSet = {'images': self.allData['images'][validationIdx:],
                            'labels': self.allData['labels'][validationIdx:]}
        else:
            self.testSet = {'images': self.allData['images'][trainIdx:],
                            'labels': self.allData['labels'][trainIdx:]}

        self.meanImage = np.mean(self.allData['images'], keepdims=True, axis=0)[0]
        # check label integrity
        for label in self.trainSet['labels']:
            if np.sum(label) != 1:
                import pdb;
                pdb.set_trace()

    def _return_label(self, label):
        if np.sum(label) != 1:
            print("ERROR!!! sum of label must be one")
            import pdb; pdb.set_trace()
        return self.hotEncodeReverse[list(label).index(1)]







class mbColorDataset:
    def __init__(self, path_to_data, image_format='tif', image_resolution=(128,128),trainSetPercentage=0.6, validationSetPercentage=0, normalize_pixel_values=True, change_dtype_to_float32=True):
        self.path = path_to_data
        self.image_format = image_format
        self.image_resolution = image_resolution
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.one_hot = False
        self._num_examples_train = 0
        self._num_examples_validation = 0
        self._num_examples_test = 0
        #####Hot_Encoding For Color####
        self.WHITE = 0; self.BLACK = 1; self.GRAY = 2; self.RED = 3; self.GREEN = 4; self.BLUE = 5
        self.YELLOW = 6; self.ORANGE = 7; self.GOLD = 8; self.OTHER = 9
        self.hotEncode = {'white': 0, 'black': 1, 'gray': 2, 'red': 3, 'green': 4, 'blue': 5,
                          'yellow': 6, 'orange': 3, 'gold': 6}
        self.hotEncodeReverse = {0: 'white', 1: 'black', 2: 'gray', 3: 'red', 4: 'green', 5: 'blue',
                                 6: 'yellow'}
        ####create dataset variables####
        self.allData = {'images': [], 'labels': []}
        ####load dataset#####
        fileList = sorted(os.listdir(self.path))
        for idx, file in enumerate(fileList):
            if (image_format in file) and (not ('txt' in file)):
                fileName = file.split('.'+image_format)[0]
                f = fileList[idx+1]
                if (fileName in f) and ('txt' in f) and (not ('mistake' in f)) and (not ('other' in f)):
                    #handle category
                    category = f.split('tif')[1].split('_')[1]
                    hotVec = np.zeros(len(self.hotEncodeReverse))
                    hotVec[self.hotEncode[category]] = 1
                    self.allData['labels'].append(hotVec.astype(np.int))
                    #handle Image
                    im = cv2.resize(cv2.imread(os.path.abspath(os.path.join(self.path, file))), self.image_resolution)
                    self.allData['images'].append(im)
                    #fileList.pop(file); fileList.pop(f)
            print ("{} loaded out of {}".format(idx, len(fileList)))
        print("Dataset loading complete")

        ####Randomly select data for train, test and validation sets####
        #Shuffle Data
        randperm = np.random.permutation(np.arange(len(self.allData['images'])))
        trainIdx = int(trainSetPercentage*len(self.allData['images']))
        self.allData['images'] = np.array(self.allData['images'])[randperm]
        if change_dtype_to_float32:
            self.allData['images'] = self.allData['images'].astype('float32')
        if normalize_pixel_values:
            self.allData['images'] /= 255.0
        self.allData['labels'] = np.array(self.allData['labels'])[randperm]
        #####create sets#####
        self.trainSet = {'images': self.allData['images'][:trainIdx],'labels': self.allData['labels'][:trainIdx]}
        self._num_examples_train = len(self.trainSet['images'])
        if validationSetPercentage != 0:
            validationIdx = int((trainSetPercentage + validationSetPercentage)*len(self.allData['images']))
            self.validationSet = {'images': self.allData['images'][trainIdx:validationIdx],
                                  'labels': self.allData['labels'][trainIdx:validationIdx]}
            self.testSet = {'images': self.allData['images'][validationIdx:], 'labels': self.allData['labels'][validationIdx:]}
        else:
            self.testSet = {'images': self.allData['images'][trainIdx:], 'labels': self.allData['labels'][trainIdx:]}

        self.meanImage = np.mean(self.allData['images'], keepdims=True, axis=0)[0]
        #check label integrity
        for label in self.trainSet['labels']:
            if np.sum(label) != 1:
                import pdb; pdb.set_trace()
        #self.check_Integrity()

    def _return_label(self, label):
        if np.sum(label) != 1:
            print("ERROR!!! sum of label must be one")
            import pdb; pdb.set_trace()
        return self.hotEncodeReverse[list(label).index(1)]


class chenColorDataset:
    hotEncode = {'white': 0, 'black': 1, 'gray': 2, 'red': 3, 'green': 4, 'blue': 5, 'yellow': 6, 'cyan': 5}
    hotEncodeReverse = {v: k for (k,v) in hotEncode.items()}
    def __init__(self, path_to_data, image_format='tif', image_resolution=(128, 128), train_set_percentage=0.6,
                 validation_set_percentage=0, normalize_pixel_values=True, change_dtype_to_float32=True,
                 gamma_correction=False, verbose=False):
        self.path = path_to_data
        self.image_format = image_format
        self.image_resolution = image_resolution
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.one_hot = False
        self._num_examples_train = 0
        self._num_examples_validation = 0
        self._num_examples_test = 0
        #####Hot_Encoding For Color####
        self.WHITE = 0; self.BLACK = 1; self.GRAY = 2; self.RED = 3; self.GREEN = 4; self.BLUE = 5
        self.YELLOW = 6; self.CYAN = 7
        self.hotEncode = chenColorDataset.hotEncode
        self.hotEncodeReverse = chenColorDataset.hotEncodeReverse
        ####create dataset variables####
        self.allData = {'images': [], 'labels': []}
        ####load dataset#####
        folderList = sorted(os.listdir(self.path))
        for idx, folder in enumerate(folderList):
            folderPath = self.path+r'/'+folder
            fileList = os.listdir(folderPath)
            for idx2, file in enumerate(fileList):
                filePath = os.path.abspath(os.path.join(folderPath, file))
                #dstPath = os.path.abspath(os.path.join(folderPath, '{sum:09d}.jpg'.format(sum=idx2)))
                #os.rename(filePath, dstPath)
                #try:
                im = cv2.resize(cv2.imread(filePath), self.image_resolution)
                hotVec = np.zeros(len(self.hotEncodeReverse))
                hotVec[self.hotEncode[folder]] = 1
                self.allData['labels'].append(hotVec.astype(np.int))
                self.allData['images'].append(im)
                #except:
                    #os.remove(dstPath)
                if verbose:
                    print ("folder {}({}/{}):\t{} images loaded out of {}".format(folder, idx, len(folderList),
                                                                                   idx2, len(fileList)))
            print("folder {}({}/{}): {} images loaded.".format(folder, idx, len(folderList), len(fileList)))
        print("Dataset loading complete")

        ####Randomly select data for train, test and validation sets####
        #Shuffle Data
        randperm = np.random.permutation(np.arange(len(self.allData['images'])))
        trainIdx = int(train_set_percentage * len(self.allData['images']))
        self.allData['images'] = np.array(self.allData['images'])[randperm]
        if change_dtype_to_float32:
            self.allData['images'] = self.allData['images'].astype('float32')
        if normalize_pixel_values:
            self.allData['images'] /= 255.0
        if gamma_correction:
            print("Starting gamma correction")
            for ind, _ in enumerate(self.allData['images']):
                if np.random.binomial(1,0.5,1)[0]:
                    gamma = np.random.uniform(0.3, 0.7)
                    self.allData['images'][ind] = cv2.pow(self.allData['images'][ind], gamma)
            print("Gamma correction ended")

        self.allData['labels'] = np.array(self.allData['labels'])[randperm]
        #####create sets#####
        self.trainSet = {'images': self.allData['images'][:trainIdx],'labels': self.allData['labels'][:trainIdx]}
        self._num_examples_train = len(self.trainSet['images'])
        if validation_set_percentage != 0:
            validationIdx = int((train_set_percentage + validation_set_percentage)*len(self.allData['images']))
            self.validationSet = {'images': self.allData['images'][trainIdx:validationIdx],
                                  'labels': self.allData['labels'][trainIdx:validationIdx]}
            self.testSet = {'images': self.allData['images'][validationIdx:], 'labels': self.allData['labels'][validationIdx:]}
        else:
            self.testSet = {'images': self.allData['images'][trainIdx:], 'labels': self.allData['labels'][trainIdx:]}

        self.meanImage = np.mean(self.allData['images'], keepdims=True, axis=0)[0]
        #check label integrity
        for label in self.trainSet['labels']:
            if np.sum(label) != 1:
                import pdb; pdb.set_trace()
        #self.check_Integrity()


    def next_batch(self, batch_size, subtractMean=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples_train)
            np.random.shuffle(perm0)
            self.trainSet['images'] = self.trainSet['images'][perm0]
            self.trainSet['labels'] = self.trainSet['labels'][perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples_train:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples_train - start
            if subtractMean:
                images_rest_part = self.trainSet['images'][start:self._num_examples_train]-self.meanImage
            else:
                images_rest_part = self.trainSet['images'][start:self._num_examples_train]
            labels_rest_part = self.trainSet['labels'][start:self._num_examples_train]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples_train)
                np.random.shuffle(perm)
                self.trainSet['images'] = self.trainSet['images'][perm]
                self.trainSet['images'] = self.trainSet['images'][perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            if subtractMean:
                images_new_part = self.trainSet['images'][start:end]-self.meanImage
            else:
                images_new_part = self.trainSet['images'][start:end]
            labels_new_part = self.trainSet['labels'][start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            if subtractMean:
                return self.trainSet['images'][start:end]-self.meanImage, self.trainSet['labels'][start:end]
            else:
                return self.trainSet['images'][start:end], self.trainSet['labels'][start:end]

    def _displayImage(self, set, imageNumber):
        print (set['labels'][imageNumber])
        cv2.imshow('image', np.array(set['images'][imageNumber]))
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()

    def _return_label(self, label):
        if np.sum(label) != 1:
            print("ERROR!!! sum of label must be one")
            import pdb; pdb.set_trace()
        return self.hotEncodeReverse[list(label).index(1)]

    def check_Integrity(self):
        for idx, image in enumerate(self.allData['images']):
            im_rs = cv2.resize(image, (360, 360))
            cv2.imshow(self._return_label(self.allData['labels'][idx]), im_rs)
            if cv2.waitKey(0) ==27:
                cv2.destroyAllWindows()



def dataSetHistogram(labels, hotencodeReverse, outf):
    hist = np.sum(labels, axis=0)
    objects = [hotencodeReverse[key] for key in range(0,len(hist))]
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, hist, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('#Samples')
    #plt.show()
    plt.savefig(outf)
    plt.close()


if __name__ == '__main__':
    dataSet = chenColorDataset(r'C:\Users\NITZANSH\Documents\TFexample\clean_database')
    dataSetHistogram(dataSet.allData['labels'], dataSet.hotEncodeReverse)