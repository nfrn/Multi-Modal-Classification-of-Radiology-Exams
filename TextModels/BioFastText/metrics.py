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
BATCH=64
LABELS = ['No Finding','Enlarged Cardiomediastinum',
              'Cardiomegaly','Airspace Opacity','Lung Lesion',
              'Edema','Consolidation','Pneumonia','Atelectasis',
              'Pneumothorax','Pleural Effusion','Pleural Other','Fracture','Support Devices']
class roc_callback(Callback):
    def __init__(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        result = self.model.predict(self.x_test)
        roc_auc = metrics.roc_auc_score(self.y_test.ravel(), result.ravel())
        print('\r Micro val_roc_auc: %s' % (str(round(roc_auc, 4))), end=100 * ' ' + '\n')

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(14):
            fpr[i], tpr[i], _ = roc_curve(self.y_test[:, i], result[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            print("Class " + str(i) + "auc = " + str(roc_auc[i]))

        macro = sum(roc_auc.values())/14
        print('\r Macro val_roc_auc: %s' % (str(round(macro, 4))), end=100 * ' ' + '\n')
        value = coverage_error(self.y_test, result)
        print('\r coverage_error: %s' % (str(round(value, 4))), end=100 * ' ' + '\n')

        value = label_ranking_loss(self.y_test, result)
        print('\r label_ranking_loss: %s' % (str(round(value, 4))), end=100 * ' ' + '\n')

        roc_auc = label_ranking_average_precision_score(self.y_test, result)
        print('\r label_ranking_average_precision_score: %s' % (str(round(roc_auc, 4))),
              end=100 * ' ' + '\n')

        return


def auc_roc(y_true, y_pred):
    score, up_opt = tf.metrics.auc(y_true, y_pred,summation_method='careful_interpolation')
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

def auc_roc2(y_true, y_pred):
    score, up_opt = tf.contrib.metrics.streaming_dynamic_auc(y_true, y_pred)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

def auc_roc3(y_true, y_pred):
    score, up_opt = tf.contrib.metrics.auc_with_confidence_intervals(y_true, y_pred)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

def precision(y_true, y_pred):
    score, up_opt = tf.metrics.precision(y_true, y_pred)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

def recall(y_true, y_pred):
    score, up_opt = tf.metrics.recall(y_true, y_pred)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

def f1(y_true, y_pred):
    score, up_opt = tf.contrib.metrics.f1_score(y_true, y_pred)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

