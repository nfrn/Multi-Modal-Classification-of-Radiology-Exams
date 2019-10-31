from __future__ import division
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.callbacks import Callback
from keras.preprocessing import image
from sklearn import metrics
from sklearn.metrics import coverage_error, roc_curve, auc
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss

PATHVAL="validate.csv"
BATCH=16
LABELS = ['No Finding','Enlarged Cardiomediastinum',
              'Cardiomegaly','Lung Opacity','Lung Lesion',
              'Edema','Consolidation','Pneumonia','Atelectasis',
              'Pneumothorax','Pleural Effusion','Pleural Other','Fracture','Support Devices']
class roc_callback(Callback):
    def __init__(self,val_gen,labels):
        self.val_gen = val_gen
        self.y = labels
    def on_train_begin(self, logs={}):
                return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        result = self.model.predict_generator(self.val_gen,
                                            steps=self.val_gen.n / BATCH,
                                            verbose=1)

        print(self.y[0])
        print(result[0])
        roc_auc = metrics.roc_auc_score(self.y.ravel(), result.ravel())
        print('\r Micro val_roc_auc: %s' % (str(round(roc_auc,4))), end=100*' '+'\n')

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(14):
            fpr[i], tpr[i], _ = roc_curve(self.y[:, i], result[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            print("Class " + str(i) + "auc = " + str(roc_auc[i]))


        value = coverage_error(self.y, result)
        print('\r coverage_error: %s' % (str(round(value,4))), end=100*' '+'\n')

        value = label_ranking_loss(self.y, result)
        print('\r label_ranking_loss: %s' % (str(round(value, 4))), end=100 * ' ' + '\n')

        roc_auc = label_ranking_average_precision_score(self.y, result)
        print('\r label_ranking_average_precision_score: %s' % (str(round(roc_auc,4))), end=100*' '+'\n')

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
