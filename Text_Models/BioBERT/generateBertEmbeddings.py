import numpy as np
import pandas as pd
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs
from allennlp.modules.elmo import batch_to_ids
MAXLEN=500
TRAIN = "../../Datasets/MIMIC-III/train.csv"
TEST = "../../Datasets/MIMIC-III/test.csv"
VAL = "../../Datasets/MIMIC-III/val.csv"


def write(filename,filename2,rows):
    config_path = "./biobert_v1.1_pubmed/bert_config.json"
    checkpoint_path = "./biobert_v1.1_pubmed/model.ckpt-1000000"
    bert = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=False,
                                              output_layer_num=4, seq_len=500)

    tokenizer= prepareTokenizer()

    df = pd.read_csv(filename, nrows=rows)
    x = df["TEXT"].values
    y = df[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
                   'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                   'Pneumothorax', 'Pleural Effusion',
                   'Pleural Other', 'Fracture', 'Support Devices']].values

    for i in range(rows):
        print(str(i) + " of " + str(rows))
        text = x[i]
        index, segment = tokenizer.encode(first=text, max_len=500)
        word_emb = bert.predict([[index], [segment]])[0]
        label = y[i]
        batch = np.array([word_emb,label])
        final_filename= "./Data/" + filename2 + "_instance_" + str(i)
        np.save(final_filename, batch)


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


def read(filename):
    array = np.load(filename)
    print(array[0])
    print(array[0].shape)
    print(array[1].shape)



if __name__ == '__main__':
    write(TRAIN,"train",169728)
    write(VAL, "val",42432)
    write(TEST, "test", 48832)
    #Test: read("./Data/train_instance_0.npy")