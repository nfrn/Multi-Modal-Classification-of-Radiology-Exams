import pickle

import keras.backend as K
import matplotlib.colors as colors
colors_list = list(colors._colors_full_map.values())
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from Utilz import ChestXRaySequence
from keras.layers import Layer
from keras_preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt

EPOCHS = 30
BATCH_SIZE = 4
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



def getData(df1,df2):
    print("Load tokenizer")
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    print("Load embedding_matrix")
    with open('embedding_matrix.pickle', 'rb') as f:
        embedding_matrix = pickle.load(f)

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

if __name__ == '__main__':
    print("Loading Model")
    '''keras.utils.generic_utils.get_custom_objects().update({'pentanh': Pentanh()})
    #model= load_model("weights-improvement-MERGED_NOTRAIN-10-0.71.hdf5",
    #                       custom_objects={'MultiplicativeLSTM': MultiplicativeLSTM,
                                           'MultiHeadAttention': MultiHeadAttention,
                                            'auc_roc': auc_roc},compile=False)
    #model.compile(loss='binary_crossentropy',
    #              optimizer='adam',
    #              metrics=['accuracy', auc_roc])
    #K.get_session().close()
    #K.set_session(tf.Session())
    #init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #K.get_session().run(init)'''

    train_df = pd.read_csv(open("./dataset/train.csv",'rU'), usecols=range(16),
                        encoding='utf-8', engine='c',dtype={'Report': str})
    val_df = pd.read_csv(open("./dataset/validate.csv",'rU'),  usecols=range(16),
                        encoding='utf-8', engine='c',dtype={'Report': str})

    x1_train, x2_train, y_train, x1_test, x2_test, y_test = getData(train_df,val_df)

    print("Prepare Generators")
    trainGen = ChestXRaySequence(x1_train, x2_train, y_train, batch_size=BATCH_SIZE)
    valGen = ChestXRaySequence(x1_test, x2_test, y_test, batch_size=BATCH_SIZE)

    print("eval")
    result = model.predict_generator(valGen,steps=valGen.n / BATCH_SIZE,verbose=1)


    np.save('data_nt10v2.npy', result)

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

    fig = plt.figure(figsize=(10, 10))
    txts = ['No findings', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    cm = plt.get_cmap('tab20')
    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1. * i / 14) for i in range(14)])

    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro {0:0.3f}'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    for i in range(14):
        ax.plot(fpr[i], tpr[i], lw=2,
                 label='{0} {1:0.3f}'
                       ''.format(txts[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    
    result = np.load("data_nt.npy")

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

    fig = plt.figure(figsize=(10, 10))
    txts = ['No findings', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    cm = plt.get_cmap('tab20')
    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1. * i / 14) for i in range(14)])

    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro {0:0.3f}'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    for i in range(14):
        ax.plot(fpr[i], tpr[i], lw=2,
                label='{0} {1:0.3f}'
                      ''.format(txts[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

    '''
    print("Start training")

    filepath = "weights-improvement-MERGED_NOTRAIN-{epoch:02d}-{val_loss:.2f}.hdf5"
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
    '''

