import random
import os

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import Sequence

import numpy as np

#https://github.com/sdcubber/Keras-Sequence-boilerplate/blob/master/Keras-Sequence.ipynb

def getImages(img):
    return img_to_array(load_img(img, color_mode="rgb", target_size=(256, 256))) / 255.

class ChestXRaySequence(Sequence):

    def __init__(self, x1,x2,y, batch_size, mode='train'):
        self.x1 = x1
        self.x2 = x2
        self.labels=y
        self.y = y
        self.n = len(self.y)
        self.batch_size = batch_size
        self.mode = mode
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.y) / float(self.batch_size)))

    def on_epoch_end(self):
        # Shuffles indexes after each epoch if in training mode
        self.indexes = range(len(self.y))
        if self.mode == 'train':
            self.indexes = random.sample(self.indexes, k=len(self.indexes))

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        labels = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]
        return labels

    def get_batch_features(self, idx):
        # Fetch a batch of inputs
        imageFeatures = np.array([getImages(im) for im in self.x2[idx * self.batch_size: (1 + idx)
                                                                                        * self.batch_size]])
        return [imageFeatures]

    def __getitem__(self, idx):
        batch_x = self.get_batch_features(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_x, batch_y

    def getTotal(self):
        labels = self.y
        imageFeatures = np.array([getImages(im) for im in self.x2])
        textReport = self.x1
        arrayReport = []
        for report in textReport:
            arrayReport.append(np.array(report))
        return [imageFeatures, np.array(arrayReport)], labels