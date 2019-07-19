import codecs
import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import torch
from allennlp.modules import Elmo
from keras import Model, Input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Bidirectional, Dense, Flatten, Layer, Lambda, GlobalMaxPooling1D, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras_bert.layers import MaskedGlobalMaxPool1D
from keras_multi_head import MultiHeadAttention
from keras_preprocessing.sequence import pad_sequences
from multiplicative_lstm import MultiplicativeLSTM
from keras import backend as K
from metrics import roc_callback
from datagenerator import DataGenerator


from CiclicLR import CyclicLR
TRAIN = "../../Datasets/MIMIC-III/train.csv"
TEST = "../../Datasets/MIMIC-III/test.csv"
VAL = "../../Datasets/MIMIC-III/val.csv"



MAX_WORDS_TEXT=500
WORD_EMBEDDINGS_SIZE=200
EPOCHS = 30
BATCH_SIZE = 2


def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.metrics.auc(y_true, y_pred)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

class Pentanh(Layer):
    def __init__(self, **kwargs):
        super(Pentanh, self).__init__(**kwargs)
        self.supports_masking = True
        self.__name__ = 'pentanh'

    def call(self, inputs):
        return K.switch(K.greater(inputs, 0), K.tanh(inputs), 0.25 * K.tanh(inputs))

    def get_config(self):
        return super(Pentanh, self).get_config()

    def compute_output_shape(self, input_shape):
        return input_shape

def loadData():
    traindf = pd.read_csv(TRAIN,nrows=169728)
    x_train = traindf["TEXT"].values
    y_train = traindf[
        ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
         'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
         'Pleural Effusion',
         'Pleural Other', 'Fracture', 'Support Devices']].values

    valdf = pd.read_csv(VAL, nrows=42432)
    x_val = valdf["TEXT"].values
    print(len(x_val))
    y_val = valdf[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
                    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                    'Pneumothorax', 'Pleural Effusion',
                    'Pleural Other', 'Fracture', 'Support Devices']].values

    testdf = pd.read_csv(TEST, nrows=48832)
    x_test = testdf["TEXT"].values
    y_test = testdf[
        ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
         'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
         'Pleural Effusion',
         'Pleural Other', 'Fracture', 'Support Devices']].values


    return x_train, y_train,x_val, y_val,x_test, y_test, testdf



def loadElmoModel():
    keras.utils.generic_utils.get_custom_objects().update({'pentanh': Pentanh()})
    input = Input((500, 1024))
    rnn_layer = Bidirectional(MultiplicativeLSTM(256,
                                                 return_sequences=True, dropout=0.2,
                                                 recurrent_dropout=0.2,
                                                 activation='pentanh',
                                                 recurrent_activation='pentanh'),
                              merge_mode='concat')(input)

    attention_layer = MultiHeadAttention(head_num=4)(rnn_layer)
    removeMask = Flatten()(attention_layer)
    final = Dense(14, activation='sigmoid')(removeMask)

    model_complete = Model(inputs=input, outputs=final)
    model_complete.compile(optimizer='adam',loss='binary_crossentropy',
                  metrics=['accuracy', auc_roc])

    print(model_complete.summary())
    return model_complete

def prepareTokenizer():
    dict_path = "./biobert_v1.1_pubmed/vocab.txt"
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    #print(token_dict)
    tokenizer = Tokenizer(token_dict)
    return tokenizer


def makePredictions():
    model = loadElmoModel()
    config_path = "./biobert_v1.1_pubmed/bert_config.json"
    checkpoint_path = "./biobert_v1.1_pubmed/model.ckpt-1000000"
    bert = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=False,
                                              output_layer_num=4, seq_len=512)
    # bert.summary()
    tokenizer = prepareTokenizer()

    valdf = pd.read_csv("validate2.csv")
    x_test = valdf["TEXT"].values
    y_test = valdf[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
                    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                    'Pneumothorax', 'Pleural Effusion',
                    'Pleural Other', 'Fracture', 'Support Devices']].values

    dataGen = DataGenerator(x_test, y_test, tokenizer, bert, tf.get_default_graph(), BATCH_SIZE,
                            True)

    model.load_weights("weights-improvement-TEXT-BERT-01-0.22.hdf5")
    results = model.predict_generator(dataGen,
                        use_multiprocessing=False,
                        verbose=1,
                        workers=1)

    np.save("bert.npy", results)

def train():
    # Load Model
    print("Load Elmo Model")
    #options_file = 'biomed_elmo_options.json'
    #weight_file = 'biomed_elmo_weights.hdf5'
    #elmo = Elmo(options_file, weight_file, 1, dropout=0)

    print("Load data")
    #x_train, y_train, x_val, y_val, x_test, y_test, testdf = loadData()
    #169728 42432 48832
    dataGen = DataGenerator("train",169728,BATCH_SIZE,True)
    dataGenval = DataGenerator("val",42432, BATCH_SIZE,False)
    dataGentest = DataGenerator("test",48832, BATCH_SIZE,False)

    print("Create Model")
    model = loadElmoModel()

    print("Preparing Training:")
    filepath = "BioElmoTextModel-{epoch:02d}-{val_loss:.2f}.hdf5"
    clr = CyclicLR(base_lr=0.00005, max_lr=0.0005, step_size=5000)
    modelckp = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,
                               mode='min')

    # es = EarlyStopping(monitor='val_acc', mode='max', min_delta=1,patience=3)
    testdf = pd.read_csv(TEST, nrows=48832)
    call_list = [modelckp, clr, roc_callback(dataGentest,testdf)]

    model.fit_generator(dataGen,
                        validation_data=dataGenval,
                        use_multiprocessing=False,
                        verbose=1,
                        epochs=EPOCHS,
                        callbacks=call_list,
                        workers=1)

    model.save_weights("BioBertTextModel.h5")

    model.save_weights("textModel-bert-.h5")

if __name__ == '__main__':
    train()