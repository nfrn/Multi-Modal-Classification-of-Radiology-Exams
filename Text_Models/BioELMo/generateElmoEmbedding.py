import numpy as np
import pandas as pd
from allennlp.modules import Elmo
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.modules.elmo import batch_to_ids
MAXLEN=500
TRAIN = "../../Datasets/MIMIC-III/train.csv"
TEST = "../../Datasets/MIMIC-III/test.csv"
VAL = "../../Datasets/MIMIC-III/val.csv"

def tokenizer(x: str):
    return [w.text for w in
            SpacyWordSplitter(language='en_core_web_sm',
                              pos_tags=False).split_words(x)[:MAXLEN]]

def write(filename,filename2,rows):
    options_file = 'biomed_elmo_options.json'
    weight_file = 'biomed_elmo_weights.hdf5'
    elmo = Elmo(options_file, weight_file, 1, dropout=0)
    elmo = elmo.to("cuda")


    df = pd.read_csv(filename, nrows=rows)
    x = df["TEXT"].values
    y = df[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
                   'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                   'Pneumothorax', 'Pleural Effusion',
                   'Pleural Other', 'Fracture', 'Support Devices']].values

    for i in range(rows):
        print(str(i) + " of " + str(rows))
        text = x[i]
        token_list=[]
        for word in tokenizer(text):
            token_list.append(word)
        for n in range(MAXLEN - len(token_list)):
            token_list.append('PAD')

        token_list = np.array([token_list])
        character_ids = batch_to_ids(token_list)
        character_ids = character_ids.to("cuda")
        word_emb = elmo(character_ids)['elmo_representations'][0]
        character_ids.to("cpu")
        word_emb = word_emb.data.cpu().numpy()
        label = y[i]
        batch = np.array([word_emb,label])
        final_filename= "./Data/" + filename2 + "_instance_" + str(i)
        np.save(final_filename, batch)

def read(filename):
    array = np.load(filename)
    print(array[0])
    print(array[0].shape)
    print(array[1].shape)



if __name__ == '__main__':
    write(TRAIN,"train",169728)
    write(VAL, "val",42432)
    write(TEST, "test", 48832)
    #Test read("./Data/test_instance_465.npy")