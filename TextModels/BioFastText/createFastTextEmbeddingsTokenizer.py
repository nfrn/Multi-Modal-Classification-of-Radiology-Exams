from gensim.models import FastText
import numpy as np
import pandas as pd
import pickle
from keras.preprocessing.text import Tokenizer
WORD_EMBEDDINGS_SIZE=200

DATATRAIN="train2.csv"
DATATEST="validate2.csv"



def loadData():
    traindf = pd.read_csv(DATATRAIN)
    x_train = traindf["TEXT"].values
    y_train = traindf[['No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Airspace Opacity',
        'Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax','Pleural Effusion',
        'Pleural Other','Fracture','Support Devices']].values

    valdf = pd.read_csv(DATATEST)
    x_test = valdf["TEXT"].values
    y_test = valdf[['No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Airspace Opacity',
        'Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax','Pleural Effusion',
        'Pleural Other','Fracture','Support Devices']].values

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    print('Average train sequence length: {}'.format(
        np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(
        np.mean(list(map(len, x_test)), dtype=int)))

    return x_train, y_train, x_test, y_test



if __name__ == '__main__':
    model = FastText.load_fasttext_format('BioWordVec_PubMed_MIMICIII_d200.vec.bin')

    x_train, y_train, x_test, y_test = loadData()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_train)
    tokenizer.fit_on_texts(y_train)
    tokenizer.fit_on_texts(x_test)
    tokenizer.fit_on_texts(y_test)


    voc_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((voc_size, WORD_EMBEDDINGS_SIZE))
    for word, i in tokenizer.word_index.items():
        embedding_matrix[i] = model.wv[word]

    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

    print("Saving tokenizer")
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saving embeddings of corpus")
    with open('embedding_matrix.pickle', 'wb') as f:
        pickle.dump(embedding_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)