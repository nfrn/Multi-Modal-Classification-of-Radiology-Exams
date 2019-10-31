import pickle
import tensorflow as tf
import keras
import keras.backend as K
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import math
import sklearn.metrics as metrics
from Utilz import ChestXRaySequence
from keras import Model
from keras_preprocessing.image import img_to_array, load_img
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Layer, Concatenate, Dense
from keras.models import load_model
from MultiHeadV2 import MultiHeadAttention
from keras_preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt
from metrics import roc_callback
from multiplicative_lstm import MultiplicativeLSTM
from CiclicLR import CyclicLR
from keras_efficientnets.custom_objects import EfficientNetDenseInitializer
from sklearn import metrics
from sklearn.metrics import coverage_error, roc_curve, auc
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss

EPOCHS = 30
BATCH_SIZE = 2
MAX_WORDS_TEXT=500

def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.metrics.auc(y_true,y_pred)

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

def mergeModels():
    print("Loading Image Model")
    imageModel = load_model("EfiNet-0X-0.32.hdf5",
                            custom_objects={'auc_roc': auc_roc,
                                            'EfficientNetConvInitializer': EfficientNetDenseInitializer}, compile=False)
    imageModel.load_weights('EfiNet-0X-0.32.hdf5')


    print("Loading Text Model")
    keras.utils.generic_utils.get_custom_objects().update({'pentanh': Pentanh()})
    textModel = load_model("weights-improvement-TEXT-0X-0.13.hdf5",
                           custom_objects={'MultiplicativeLSTM': MultiplicativeLSTM,
                                           'MultiHeadAttention': MultiHeadAttention},
                           compile=False)
    textModel.load_weights('weights-improvement-TEXT-0X-0.13.hdf5')


    print("Merging Models")

    img_final_layer = imageModel.get_layer('global_average_pooling2d_1').output
    txt_final_layer = textModel.get_layer('flatten_1').output

    joinedFeatures_layer = Concatenate(name='Final_Merge')([img_final_layer, txt_final_layer])

    final = Dense(14, activation='sigmoid')(joinedFeatures_layer)

    model = Model(inputs=[imageModel.input, textModel.input], outputs=final)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', auc_roc])

    return model

def mergeModelsRandomWeights(model):
    model2 = mergeModels()
    json_string = model2.to_json()
    from keras.models import model_from_json
    model = model_from_json(json_string,custom_objects={'MultiplicativeLSTM': MultiplicativeLSTM,
                                            'MultiHeadAttention': MultiHeadAttention,
                                            'auc_roc': auc_roc,
                                            'EfficientNetConvInitializer': EfficientNetDenseInitializer})
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', auc_roc])

    K.get_session().close()
    K.set_session(tf.Session())
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    K.get_session().run(init)
    return model

def getData(df1,df2,df3):
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
        filename = "." + path[1:]
        x2_train[idx] = filename

    y_train = df1[['No findings', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
                  'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                  'Pneumothorax',
                  'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']].values

    print("Preparing val data")
    x1_val = df2["Report"].values
    x1_val = tokenizer.texts_to_sequences(x1_val)
    x1_val = pad_sequences(x1_val, maxlen=MAX_WORDS_TEXT, padding='post')
    x2_val = df2["Img"].values
    for idx, path in enumerate(x2_val):
        filename = "." + path[1:]
        x2_val[idx] = filename
    y_val = df2[['No findings', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis','Pneumothorax',
                 'Pleural Effusion', 'Pleural Other','Fracture', 'Support Devices']].values


    print("Preparing val data")
    x1_test = df3["Report"].values
    x1_test = tokenizer.texts_to_sequences(x1_test)
    x1_test = pad_sequences(x1_test, maxlen=MAX_WORDS_TEXT, padding='post')
    x2_test = df3["Img"].values
    for idx, path in enumerate(x2_test):
        filename = "." + path[1:]
        x2_test[idx] = filename
    y_test = df3[['No findings', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis','Pneumothorax',
                 'Pleural Effusion', 'Pleural Other','Fracture', 'Support Devices']].values

    return x1_train, x2_train, y_train, x1_val, x2_val, y_val, x1_test, x2_test, y_test

def makePredictions(model):

    train_df = pd.read_csv(open("./train.csv",'rU'), usecols=range(16),
                        encoding='utf-8', engine='c',dtype={'Report': str})
    val_df = pd.read_csv(open("./validate.csv",'rU'),  usecols=range(16),
                        encoding='utf-8', engine='c',dtype={'Report': str})
    test_df = pd.read_csv(open("./test.csv",'rU'),  usecols=range(16),
                        encoding='utf-8', engine='c',dtype={'Report': str})

    x1_train, x2_train, y_train, x1_val, x2_val, y_val,x1_test, x2_test, y_test = getData(train_df,val_df,test_df)
    testGen = ChestXRaySequence(x1_test, x2_test, y_test, batch_size=BATCH_SIZE)

    predictions = model.predict_generator(testGen,steps=testGen.n / BATCH_SIZE,verbose=1)
    array = np.array([predictions, testgenerator.labels])
    np.save('eff_ft_no.npy', array)

def train(model):

    train_df = pd.read_csv(open("./train.csv",'rU'), usecols=range(16),
                        encoding='utf-8', engine='c',dtype={'Report': str})
    val_df = pd.read_csv(open("./validate.csv",'rU'),  usecols=range(16),
                        encoding='utf-8', engine='c',dtype={'Report': str})
    test_df = pd.read_csv(open("./test.csv",'rU'),  usecols=range(16),
                        encoding='utf-8', engine='c',dtype={'Report': str})

    x1_train, x2_train, y_train, x1_val, x2_val, y_val,x1_test, x2_test, y_test = getData(train_df,val_df,test_df)

    print("Prepare Generators")
    trainGen = ChestXRaySequence(x1_train, x2_train, y_train, batch_size=BATCH_SIZE)
    valGen = ChestXRaySequence(x1_val, x2_val, y_val, batch_size=BATCH_SIZE)
    testGen = ChestXRaySequence(x1_test, x2_test, y_test, batch_size=BATCH_SIZE)

    print("Start training")

    filepath = "weights-improvement-MERGED-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,
                                 mode='min')
    clr = CyclicLR(base_lr=0.0001, max_lr=0.0006, step_size=2000.)
    callbacks_list = [checkpoint,clr,
                      roc_callback(testGen, np.array(testGen.labels))]


    model.fit_generator(generator=trainGen,
                        validation_data=valGen,
                        use_multiprocessing=False,
                        verbose=1,
                        epochs=EPOCHS,
                        callbacks=callbacks_list,
                        workers=6)

    model.save("merged.h5")

def visualizeAttention(filename,sentence,weights1,weights2,reverse_word_map):
    #Adapted from https://github.com/qq345736500/sarcasm/blob/3e55cf4f153efd7d01caa553b05af883a268f418/src/visualize_tf_attention.py
    print(sentence)
    with open(filename, "w") as html_file:
        html_file.write('<!DOCTYPE html>\n')
        html_file.write('<html>\n'
                        '<font size="5">\n'
                        '<head>\n'
                        '<meta charset="UTF-8">\n'
                        '<font size="7"><b>'
                        'Report Attention Weights Visualization</b></font size>'
                        '<br><br>'
                        '</head>\n')
        html_file.write('<body style="text-align:justify;">\n')
        print("Preparing the vizualization for the attention coefficients...")

        for word, alpha1, alpha2 in zip(sentence[0], weights1 / weights1.max(),weights2 /
                weights2.max()):
            word = reverse_word_map.get(word)

            print(alpha1)

            colorRGBA2= [51,102,153,alpha1*255]
            colorRGBA1= [153,255,153,alpha2*255]
            alpha = (255 - ((255 - colorRGBA1[3]) * (255 - colorRGBA2[3]) / 255)) / 255
            red = (colorRGBA1[0] * (255 - colorRGBA2[3]) + colorRGBA2[0] * colorRGBA2[3]) / 255
            green = (colorRGBA1[1] * (255 - colorRGBA2[3]) + colorRGBA2[1] * colorRGBA2[3]) / 255
            blue = (colorRGBA1[2] * (255 - colorRGBA2[3]) + colorRGBA2[2] * colorRGBA2[3]) / 255

            print([red,green,blue,alpha])

            if word != None:
                html_file.write('<font style="background: rgb(%f,%f,%f,%f)">%s</font>\n' % (
                    red,green,blue,alpha, word))

        html_file.write('</body></font></html>')
    print('\nA visualization for the attention coefficients is now available in attention_vis.html')

def viewTextWeights(model):
    sentence = """COMPARISON XXXX  INDICATION XXXX-year-old male with end-stage renal disease on hemodialysis  FINDINGS The heart size is mildly enlarged. There is tortuosity of the thoracic aorta. No focal airspace consolidation, pleural effusions or pneumothorax. No acute bony abnormalities.  IMPRESSION Cardiomegaly without acute pulmonary findings.    print("Load tokenizer")"""
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    print("Preparing train data")
    x1_train = tokenizer.texts_to_sequences([sentence])
    x1_train = pad_sequences(x1_train, maxlen=MAX_WORDS_TEXT, padding='post')

    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    test = np.array(x1_train)

    image3 = img_to_array(load_img("./images/CXR3557_IM-1742-0001-0002.png",
                                   color_mode="rgb", target_size=(256, 256))) / 255.

    image2 = np.array([image3])
    print(np.shape(image2))
    print(np.shape(test))
    batch = [image2, test]
    # print(np.shape(batch))

    input = model.get_layer('Input').input
    layer = model.get_layer('multi_head_attention_1')

    print(model.input)
    predict = model.predict(batch)
    print(predict)
    getAttentionValues = K.function([input], [layer.a])
    a = getAttentionValues([test])

    a = np.reshape(a, (4, 500, 500))

    print(sentence)

    attentions = a[0].mean(0)
    attentionHead1 = ((attentions - np.amin(attentions)) * (1 / ((np.amax(attentions)) - np.amin(
        attentions))) * 255).astype('uint8')
    print(attentionHead1)

    attentions = a[1].mean(0)
    attentionHead2 = ((attentions - np.amin(attentions)) * (1 / ((np.amax(attentions)) - np.amin(
        attentions))) * 255).astype('uint8')
    print(attentionHead2)

    attentions = a[2].mean(0)
    attentionHead3 = ((attentions - np.amin(attentions)) * (1 / ((np.amax(attentions)) - np.amin(
        attentions))) * 255).astype('uint8')
    print(attentionHead3)

    attentions = a[3].mean(0)
    attentionHead4 = ((attentions - np.amin(attentions)) * (1 / ((np.amax(attentions)) - np.amin(
        attentions))) * 255).astype('uint8')
    print(attentionHead4)

    np.save("ah1.npy", attentionHead1)
    np.save("ah2.npy", attentionHead2)
    np.save("ah3.npy", attentionHead3)
    np.save("ah4.npy", attentionHead4)

    visualizeAttention("./2attention1_2.html", x1_train, attentionHead1, attentionHead2,
                       reverse_word_map)
    visualizeAttention("./2attention1_3.html", x1_train, attentionHead1, attentionHead3,
                       reverse_word_map)
    visualizeAttention("./2attention1_4.html", x1_train, attentionHead1, attentionHead4,
                       reverse_word_map)
    visualizeAttention("./2attention2_3.html", x1_train, attentionHead2, attentionHead3,
                       reverse_word_map)
    visualizeAttention("./2attention2_4.html", x1_train, attentionHead2, attentionHead4,
                       reverse_word_map)
    visualizeAttention("./2attention3_4.html", x1_train, attentionHead3, attentionHead4,
                       reverse_word_map)

def viewGradCam(model):
    image = img_to_array(load_img("./images/CXR2484_IM-1012-1001.png",
                                  color_mode="rgb", target_size=(256, 256))) / 255.

    image3 = img_to_array(load_img("./images/CXR2484_IM-1012-1001.png",
                                   color_mode="rgb", target_size=(256, 256))) / 255.

    image2 = np.expand_dims(image, axis=0)

    report = "COMPARISON None.  INDICATION Dizzy. Unable to XXXX.  FINDINGS Frontal and lateral views of the chest with overlying external cardiac monitor leads show normal size and configuration of the cardiac silhouette. There are scattered nodular opacities, XXXX calcified granulomas. No XXXX focal airspace consolidation or pleural effusion.  IMPRESSION No acute or active cardiac, pulmonary or pleural disease. Probable previous granulomatous disease. "

    print("Load tokenizer")
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    x1_train = tokenizer.texts_to_sequences([report])
    x1_train = pad_sequences(x1_train, maxlen=MAX_WORDS_TEXT, padding='post')

    #layer= Choose the one you want

    predicted = model.predict([image2,x1_train])
    print(predicted)
    predicted[predicted >= 0.5] = 1
    predicted[predicted < 0.5] = 0
    print(predicted)
    predicted = np.array(predicted)
    indexes = np.where(predicted[0] == 1)[0]
    print(indexes)
    plt.imshow(image3)
    plt.show()
    for index in indexes:
        grads = visualize_cam(model, layer, index, image2)
        plt.imshow(image3)
        plt.imshow(grads, cmap='jet', alpha=0.4)
        plt.show()

if __name__ == '__main__':
    model = mergeModels()
    # model = mergeModelsRandomWeights(model):
    train(model)
    #model.load_weights(filename)
    makePredictions(model)
    viewTextWeights(model)
    viewGradCam(model)