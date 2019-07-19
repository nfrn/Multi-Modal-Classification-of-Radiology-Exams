import math
import threading

import keras
import numpy as np

MAXLEN=500


class DataGenerator(keras.utils.Sequence):
    def __init__(self, mode,size, batch_size=1,shuffle=False):
        self.batch_size = batch_size
        self.mode = mode
        self.lock = threading.Lock()
        self.n = size
        self.indexes = np.arange(self.n)
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(self.n / float(self.batch_size))

    def __getitem__(self, index):
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batchX=[]
        batchY=[]
        for k in batch_indexs:
            filename = "./Data/"+self.mode + "_instance_"+str(k)+".npy"
            instance = np.load(filename)
            X = instance[0][0]
            Y = instance[1]
            batchX.append(X)
            batchY.append(Y)

        batchX = np.array(batchX)
        batchY = np.array(batchY)
        print(batchX.shape)
        print(batchY.shape)
        return batchX,batchY

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


