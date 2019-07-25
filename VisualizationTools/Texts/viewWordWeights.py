import pickle
import tqdm
import keras
import keras.backend as K
import numpy as np
from keras import Model
from keras.layers import Layer, Concatenate, Dense
from keras.models import load_model
from MultiHeadV2 import MultiHeadAttention
from keras_preprocessing.sequence import pad_sequences
from multiplicative_lstm import MultiplicativeLSTM
import tensorflow as tf
EPOCHS = 30
BATCH_SIZE = 2
MAX_WORDS_TEXT=500

def auc_roc(y_true, y_pred):
    score, up_opt = tf.metrics.auc(y_true, y_pred,summation_method='careful_interpolation')
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

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



def visualizeAttention(filename,sentence,weights,reverse_word_map):
    #Adapted from https://github.com/qq345736500/sarcasm/blob/3e55cf4f153efd7d01caa553b05af883a268f418/src/visualize_tf_attention.py
    print(sentence)
    print(weights)
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
        html_file.write('<body>\n')
        print("Preparing the vizualization for the attention coefficients...")

        for word, alpha in zip(sentence[0], weights / weights.max()):
            word = reverse_word_map.get(word)
            if word != None:
                html_file.write('<font style="background: rgba(255, 0, 0, %f)">%s</font>\n' % (alpha, word))

        html_file.write('</body></font></html>')
    print('\nA visualization for the attention coefficients is now available in attention_vis.html')


if __name__ == '__main__':


    sentence= "COMPARISON None  INDICATION: Hypertension FINDINGS Density in the left upper lung on PA XXXX XXXX" \
              " represents superimposed bony and vascular structures. There is calcification of the first rib" \
              " costicartilage junction which XXXX contributes to this appearance. The lungs otherwise appear clear." \
              " The heart and pulmonary XXXX appear normal. In the pleural spaces are clear. The mediastinal contour" \
              " is normal. There are degenerative changes of thoracic spine. There is an electronic cardiac device" \
              " overlying the left chest wall with intact distal leads in the right heart.  IMPRESSION 1. Irregular" \
              " density in the left upper lung on PA XXXX, XXXX artifact related to superimposed vascular bony" \
              " structures. Chest fluoroscopy or XXXX would confirm this 2.Otherwise, no acute cardiopulmonary disease."
    print("Load tokenizer")
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    print("Preparing train data")
    x1_train = tokenizer.texts_to_sequences([sentence])
    x1_train = pad_sequences(x1_train, maxlen=MAX_WORDS_TEXT, padding='post')

    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

    print(x1_train)

    model = mergeModels()
    model.load_weights("weights-improvement-MERGED-11-0.11.hdf5")
    input = model.get_layer('Input').input
    layer = model.get_layer('multi_head_attention_1')

    test = np.reshape(x1_train[0], (1, 500))

    getAttentionValues = K.function([input], [layer.a])
    a = getAttentionValues([test])

    test = np.reshape(x1_train[0], (500))

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

    visualizeAttention("./attention1.html", x1_train, attentionHead1, reverse_word_map)
    visualizeAttention("./attention2.html", x1_train, attentionHead2, reverse_word_map)
    visualizeAttention("./attention3.html", x1_train, attentionHead3, reverse_word_map)
    visualizeAttention("./attention4.html", x1_train, attentionHead4, reverse_word_map)


