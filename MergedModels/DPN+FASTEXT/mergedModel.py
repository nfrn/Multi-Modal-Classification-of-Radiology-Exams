import os
import pickle
import time

import keras
import keras.backend as K
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from Utilz import ChestXRaySequence
from keras import Model
from keras import activations
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Layer, Concatenate, Dense
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras_multi_head import MultiHeadAttention
from keras_preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt
from metrics import roc_callback, coverage_error_callback, \
    label_ranking_average_precision_score_callback, label_ranking_loss_callback, \
    auc_roc, auc_roc2, auc_roc3, precision, recall, f1
from multiplicative_lstm import MultiplicativeLSTM
from vis.utils import utils
from vis.visualization import visualize_cam

EPOCHS = 30
BATCH_SIZE = 2
MAX_WORDS_TEXT=500

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

def mergeModels():
    print("Loading Image Model")
    imageModel = load_model("weights-improvement-V2-BINARY-14-0.38.hdf5",
                            custom_objects={'auc_roc': auc_roc}, compile=False)
    # print(imageModel.summary())

    print("Loading Text Model")
    keras.utils.generic_utils.get_custom_objects().update({'pentanh': Pentanh()})
    textModel = load_model("weights-improvement-TEXT-09-0.13.hdf5",
                           custom_objects={'MultiplicativeLSTM': MultiplicativeLSTM,
                                           'MultiHeadAttention': MultiHeadAttention},
                           compile=False)

    # print(textModel.summary())
    print("Merging Models")

    img_final_layer = imageModel.get_layer('lambda_1029').output
    txt_final_layer = textModel.get_layer('flatten_1').output

    joinedFeatures_layer = Concatenate(name='Final_Merge')([img_final_layer, txt_final_layer])

    final = Dense(14, activation='sigmoid')(joinedFeatures_layer)

    model = Model(inputs=[imageModel.input, textModel.input], outputs=final)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', auc_roc])

    return model

def getData(df1,df2):
    print("Load tokenizer")
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    print("Preparing train data")
    x1_train = df1["Report"].values
    print(x1_train[0])
    x1_train = tokenizer.texts_to_sequences(x1_train)
    x1_train = pad_sequences(x1_train, maxlen=MAX_WORDS_TEXT, padding='post')
    x2_train = df1["Img"].values
    for idx, path in enumerate(x2_train):
        filename = "./dataset" + path[1:]
        x2_train[idx] = filename

    y_train = df1[['No findings', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
                  'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                  'Pneumothorax',
                  'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']].values

    print("Preparing val data")
    x1_test = df2["Report"].values
    x1_test = tokenizer.texts_to_sequences(x1_test)
    x1_test = pad_sequences(x1_test, maxlen=MAX_WORDS_TEXT, padding='post')
    x2_test = df2["Img"].values
    for idx, path in enumerate(x2_test):
        filename = "./dataset" + path[1:]
        x2_test[idx] = filename
    y_test = df2[['No findings', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis','Pneumothorax',
                 'Pleural Effusion', 'Pleural Other','Fracture', 'Support Devices']].values

    return x1_train, x2_train, y_train, x1_test, x2_test, y_test


def eval(valGen):

    model = mergeModels()
    model.load_weights("weights-improvement-MERGED-11-0.11.hdf5")
    result = model.predict_generator(valGen,steps=valGen.n / BATCH_SIZE,verbose=1)
    print(result)
    #np.save('data_nt27.npy', result)
    x, y = valGen.getTotal()
    result = np.load("data.npy")

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(14):
        output = np.count_nonzero(np.count_nonzero(y[:, i] == 1))
        print(len(y[:, i]))
        fpr[i], tpr[i], _ = metrics.roc_curve(y[:, i], result[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(14)]))

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y.ravel(), result.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(14):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average {0:0.3f}'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    colors_list = list(colors._colors_full_map.values())
    class_list = ['No findings', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
                  'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                  'Pneumothorax',
                  'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    for i in range(14):
        plt.plot(fpr[i], tpr[i], color=colors_list[i*3], lw=1,
                 label='{0} {1:0.3f}'
                       ''.format(class_list[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Image and Text Classification on OpenI with pretrain weights')
    plt.legend(loc="lower right")
    plt.show()



def train():
    model = mergeModels()
    model.load_weights("weights-improvement-MERGED_NOTRAIN-27-0.27.hdf5")
    train_df = pd.read_csv(open("./dataset/train.csv",'rU'), usecols=range(16),
                        encoding='utf-8', engine='c',dtype={'Report': str})
    val_df = pd.read_csv(open("./dataset/validate.csv",'rU'),  usecols=range(16),
                        encoding='utf-8', engine='c',dtype={'Report': str})

    x1_train, x2_train, y_train, x1_test, x2_test, y_test = getData(train_df,val_df)

    print("Prepare Generators")
    trainGen = ChestXRaySequence(x1_train, x2_train, y_train, batch_size=BATCH_SIZE)
    valGen = ChestXRaySequence(x1_test, x2_test, y_test, batch_size=BATCH_SIZE)


    print("Start training")

    filepath = "weights-improvement-MERGED-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,
                                 mode='min')

    callbacks_list = [checkpoint, ReduceLROnPlateau(),
                      roc_callback(valGen)]

    model.save("mode_final_pretrain.h5")

    model.fit_generator(generator=trainGen,
                        validation_data=valGen,
                        use_multiprocessing=False,
                        verbose=1,
                        epochs=EPOCHS,
                        callbacks=callbacks_list,
                        workers=6)

    model.save("merged.h5")

if __name__ == '__main__':
    model = mergeModels()
    model.load_weights("weights-improvement-MERGED-11-0.11.hdf5")
    train_df = pd.read_csv(open("./dataset/train.csv", 'rU'), usecols=range(16),
                           encoding='utf-8', engine='c', dtype={'Report': str})
    val_df = pd.read_csv(open("./dataset/validate.csv", 'rU'), usecols=range(16),
                         encoding='utf-8', engine='c', dtype={'Report': str})

    x1_train, x2_train, y_train, x1_test, x2_test, y_test = getData(train_df, val_df)

    print(x1_train[0])
    print(x2_train[0])
    valGen = ChestXRaySequence(x1_test[:4], x2_test[:4], y_test[:4], batch_size=BATCH_SIZE)
    result = model.predict_generator(valGen,steps=valGen.n/BATCH_SIZE,verbose=1)
    #print(result)
    a,labels = valGen.getTotal()
    for idx, result in enumerate(result):
        print("Test:"+str(idx))
        print(result)
        print(labels[idx])
    #pred = np.load("data.npy")
    #print(pred)
    model = mergeModels()
    model.load_weights("weights-improvement-MERGED-11-0.11.hdf5")
    #print(model.summary())

    text = ["COMPARISON None  INDICATION: Hypertension FINDINGS Density in the left upper lung on PA XXXX XXXX represents superimposed bony and vascular structures. There is calcification of the first rib costicartilage junction which XXXX contributes to this appearance. The lungs otherwise appear clear. The heart and pulmonary XXXX appear normal. In the pleural spaces are clear. The mediastinal contour is normal. There are degenerative changes of thoracic spine. There is an electronic cardiac device overlying the left chest wall with intact distal leads in the right heart.  IMPRESSION 1. Irregular density in the left upper lung on PA XXXX, XXXX artifact related to superimposed vascular bony structures. Chest fluoroscopy or XXXX would confirm this 2.Otherwise, no acute cardiopulmonary disease."]

    print("Load tokenizer")
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    print("Preparing train data")
    x1_train = tokenizer.texts_to_sequences(text)
    x1_train = pad_sequences(x1_train, maxlen=MAX_WORDS_TEXT, padding='post')
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

    print(x1_train)

    output = model.get_layer('multi_head_attention_1').output
    input = model.get_layer('Input').input

    layer = model.get_layer('multi_head_attention_1')

    test = np.reshape(x1_train[0], (1, 500))

    getAttentionValues = K.function([input], [layer.a])
    a = getAttentionValues([test])

    test = np.reshape(x1_train[0], (500))
    sentence= []
    for x in test:
        if x !=0:
            sentence.append(reverse_word_map[x])

    np.save("sentence.npy", sentence)

    a = np.reshape(a,(4,500,500))

    print(sentence)

    attentions = a[0].mean(0)
    attentionHead1 = ((attentions - np.amin(attentions)) * (1 / ((np.amax(attentions)) - np.amin(
        attentions))) * 255).astype('uint8')
    print(attentionHead1)

    np.save("head1.npy", attentionHead1)

    attentions = a[1].mean(0)
    attentionHead1 = ((attentions - np.amin(attentions)) * (1 / ((np.amax(attentions)) - np.amin(
        attentions))) * 255).astype('uint8')
    print(attentionHead1)
    np.save("head2.npy", attentionHead1)

    attentions = a[2].mean(0)
    attentionHead1 = ((attentions - np.amin(attentions)) * (1 / ((np.amax(attentions)) - np.amin(
        attentions))) * 255).astype('uint8')
    print(attentionHead1)
    np.save("head3.npy", attentionHead1)

    attentions = a[3].mean(0)
    attentionHead1 = ((attentions - np.amin(attentions)) * (1 / ((np.amax(attentions)) - np.amin(
        attentions))) * 255).astype('uint8')
    print(attentionHead1)
    np.save("head4.npy", attentionHead1)
