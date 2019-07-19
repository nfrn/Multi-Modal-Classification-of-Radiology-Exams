import math

import keras
import numpy as np
import tensorflow as tf

class DataGenerator(keras.utils.Sequence):
    def __init__(self, dataX, dataY, tokenizer, bert_model,tf_graph, batch_size=1, shuffle=True):
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.batch_size = batch_size
        self.graph= tf_graph
        self.dataX = dataX
        self.n = len(dataX)
        self.dataY = dataY
        assert (len(self.dataX) == len(self.dataY))
        self.indexes = np.arange(len(self.dataX))
        self.shuffle = shuffle
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.dataX) / float(self.batch_size))

    def __getitem__(self, index):
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_X = [self.dataX[k] for k in batch_indexs]
        batch_Y = [self.dataY[k] for k in batch_indexs]

        X, y = self.data_generation(batch_X, batch_Y)
        return X, y

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_X, batch_Y):
        y = []

        we=[]
        with self.graph.as_default():
            for text in batch_X:
                common_seg_input = np.zeros((1, 500), dtype=np.float32)
                index, segment = self.tokenizer.encode(first=text, max_len=500)
                ww = self.bert_model.predict([[index],[segment]])[0]

                we.append(ww)

            #we = np.squeeze(we, 1)
            we = np.array(we)
            batch_Y = np.array(batch_Y)
            print(np.shape(batch_Y))
            print(np.shape(we))

            return we,batch_Y