import glob
import pandas as pd
import xmltodict
from collections import Counter
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import os.path
import numpy as np

from keras_preprocessing.text import Tokenizer
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix

""" 7,470 chest x-rays with 3,955 reports
    FRONT IMAGE = """


def getText(file):
    stringa = file.find("<Abstract>")
    stringb = file.find("</Abstract>")
    if stringa == -1 or stringb == -1:
        print("No abstract")
        return "NO ABSTRACT"
    all = file[stringa:stringb]
    all = re.sub("<Abstract>", "", all)
    all = re.sub("<AbstractText Label=", "", all)
    all = re.sub("</AbstractText>", "", all)
    all = re.sub(" +", " ", all)
    all = re.sub('\"', "", all)
    all = re.sub('>', " ", all)
    all = re.sub('\n', "", all)
    return all

def getLabels(doc):
    # GET LABELS
    value = []
    for idx2, node in enumerate(doc.getElementsByTagName('MeSH')):
        for elem in node.childNodes:
            string = elem.toxml()
            string = re.sub("<automatic>", "", string)
            string = re.sub("</automatic>", "", string)
            string = re.sub("<major>", "", string)
            string = re.sub("</major>", "", string)
            string = re.sub("\n", "", string)
            if "  " not in string:
                value.append(string)

    return value

def labelsprocessing():
    df = pd.read_csv("dataset.csv")

    for idx,x in enumerate(df["Labels"]):
        x = re.sub("","",x)
        x = re.sub("\[", "", x)
        x = re.sub("]", "", x)
        x = re.sub("/", " ", x)
        x = re.sub(",", " ", x)
        x = re.sub("'", " ", x)
        x = re.sub("{ }+ ", " ", x)
        x = re.sub(' +', ' ',x)
        words = word_tokenize(x)
        words = [word.lower() for word in words]
        df.set_value(idx, "Labels", " ".join(words))

def getImage(file):
    starts = [match.start() for match in re.finditer(re.escape("<parentImage id="), file)]
    ends = [match.start() for match in re.finditer(re.escape("<figureId>"), file)]

    filenames = []
    for idx, val in enumerate(starts):
        stringa = val
        stringb = ends[idx]
        if val == -1 or stringb == -1:
            continue
        filename = "./images/" + file[stringa + 17:stringb - 23] + ".png"
        filenames.append(filename)

    for filename in filenames:
        if os.path.isfile(filename):
            return filename

    return "NO"
def xmlToDF():
    df = pd.DataFrame(columns=["Img","Labels","Report"])
    counter=1
    for idx, file in enumerate(glob.glob("./ecgen-radiology/*.xml")):
        doc = minidom.parse(file)
        file = doc.toxml()

        labels = getLabels(doc)
        text = getText(file)
        img = getImage(file)

        if "NO" in img:
            continue
        df.set_value(idx, "Img", img)
        df.set_value(idx, "Labels", labels)
        df.set_value(idx, "Report", text)

        print(counter)
        counter+=1

    df.to_csv("dataset.csv")

def stratify():
    df = pd.read_csv("dataset4.csv", usecols=['Img','Report', 'No findings', 'Enlarged '
                                                                             'Cardiomediastinum',
                                           'Cardiomegaly', 'Airspace Opacity','Lung Lesion',
                                           'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                                           'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                                           'Fracture', 'Support Devices'])

    totalX = df[['Img','Report']].values
    totalY = df[['No findings', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis','Pneumothorax',
                 'Pleural Effusion', 'Pleural Other','Fracture', 'Support Devices']].values

    print(totalX.shape)
    print(totalY.shape)

    totalX = np.expand_dims(totalX, axis=1)

    print("PRE ITERATIVE")
    X_train, y_train, X_test, y_test = iterative_train_test_split(totalX, totalY, 0.2)

    print("COMBINATION")
    df = pd.DataFrame({
        'train': Counter(
            str(combination) for row in get_combination_wise_output_matrix(y_train, order=2)
            for
            combination in row),
        'test': Counter(
            str(combination) for row in get_combination_wise_output_matrix(y_test, order=2) for
            combination in row)
    }).T.fillna(0.0)
    print(df.to_string())

    X_train = np.squeeze(X_train, axis=1)
    X_test = np.squeeze(X_test, axis=1)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)


    print("WRITING Train")

    dfTotal2 = pd.DataFrame(columns=['Img','Report', 'No findings', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis','Pneumothorax',
                 'Pleural Effusion', 'Pleural Other','Fracture', 'Support Devices'])
    print(dfTotal2.shape)
    dfTotal2[['Img','Report']] = pd.DataFrame(X_train)
    dfTotal2[['No findings', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis','Pneumothorax',
                 'Pleural Effusion', 'Pleural Other','Fracture', 'Support Devices']] = y_train

    with open("train.csv", mode='w', newline='\n') as f:
        dfTotal2.to_csv(f, sep=",", float_format='%.2f', index=False, line_terminator='\n',
                    encoding='utf-8')


    print("WRITING Test")

    dfTotal2 = pd.DataFrame(columns=['Img','Report', 'No findings', 'Enlarged Cardiomediastinum',
                                     'Cardiomegaly', 'Airspace Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis','Pneumothorax',
                 'Pleural Effusion', 'Pleural Other','Fracture', 'Support Devices'])
    dfTotal2[['Img','Report']] = pd.DataFrame(X_test)
    dfTotal2[['No findings', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis','Pneumothorax',
                 'Pleural Effusion', 'Pleural Other','Fracture', 'Support Devices']] = y_test
    dfTotal2.to_csv("test.csv")
    with open("test.csv", mode='w', newline='\n') as f:
        dfTotal2.to_csv(f, sep=",", float_format='%.2f', index=False, line_terminator='\n',
                    encoding='utf-8')

def labels():
    df = pd.read_csv("dataset2.csv")
    ITEMS = {'No findings':['normal'],
             'Enlarged Cardiomediastinum': ['enlarged mediastinum'],
             'Cardiomegaly': ['cardiomegaly'],
             'Airspace Opacity': ['opacity'],
             'Lung Lesion': ['lung'],
             'Edema': ['edema','edemas'],
             'Consolidation': ['consolidation'],
             'Pneumonia': ['pneumonia'],
             'Atelectasis': ['atelectasis'],
             'Pneumothorax': ['pneumothorax','hydropneumothorax'],
             'Pleural Effusion': ['pleural effusion','pleural effusions'],
             'Pleural Other': ["pleural thickening",'pleural diseases'],
             'Fracture': ['fracture','fractures'],
             'Support Devices': ['medical device']}


    print(len(df.index))
    for label in ITEMS.keys():
        df[label] = 0
    for label in ITEMS.keys():
        print(label)
        for code in ITEMS.get(label):
            if code == "normal":
                idx = df.index[df['Labels'] == "normal"]
                df.loc[idx,label] = 1
            else:
                df.loc[df['Labels'].str.contains(code), label] = 1
    df.to_csv("dataset3.csv")



def checkNoFinding():
    df = pd.read_csv('dataset3.csv',
                     usecols=['Img','Report', 'No findings', 'Enlarged Cardiomediastinum',
                              'Cardiomegaly', 'Airspace Opacity',
                              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                              'Atelectasis',
                              'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                              'Support Devices'])

    df2 = pd.read_csv('dataset3.csv',
                      usecols=['No findings', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                               'Airspace Opacity',
                               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                               'Atelectasis',
                               'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                               'Support Devices'])

    for idx,x in enumerate(df2.sum(axis=1)):
        if x ==0:
            print("changed index:" + str(idx))
            df.loc[idx, 'No findings'] = 1

    df.to_csv('dataset4.csv')



if __name__ == '__main__':
    labels()
    checkNoFinding()
    stratify()