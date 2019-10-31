import math
import multiprocessing.dummy as mp
from keras.utils import Sequence
import numpy as np
import threading

def read_data(path):
    data = np.load(path, allow_pickle=True)
    return data


def merge(XL,YL,X,Y):
    XL.append(X)
    YL.append(Y)
    return XL,YL


class DataGenerator(Sequence):
    def __init__(self, mode,size, batch_size=1,shuffle=False):
        self.batch_size = batch_size
        self.mode = mode
        self.n = size
        self.indexes = np.arange(self.n)
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(self.n / float(self.batch_size))

    def __getitem__(self, index):
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        filenames=[]
        for k in batch_indexs:
            filename = "./Data/"+self.mode + "_instance_"+str(k)+".npy"
            filenames.append(filename)

        p=mp.Pool(8)
        results = [p.map(read_data, filenames)]
        p.close()
        p.join()

        batchX=[]
        batchY=[]
        for idx,result in enumerate(results[0]):
            batchX.append(result[0])
            batchY.append(result[1])

        batchX = np.array(batchX)
        batchY = np.array(batchY)
        return batchX, batchY

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


