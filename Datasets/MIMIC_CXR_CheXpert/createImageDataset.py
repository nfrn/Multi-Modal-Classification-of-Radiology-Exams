import numpy as np
import pandas as pd
from collections import Counter
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix
from skmultilearn.model_selection import iterative_train_test_split


def mergeChexPert():
    df = pd.read_csv('train_chexpert.csv')
    df2 = pd.read_csv('test_chexpert.csv')

    result = pd.concat([df,df2])
    result = result.loc[result['Frontal/Lateral'] == "Frontal"]
    result = result.fillna(0)
    result = result.replace(-1.0,1.0)

    result.to_csv("total_chexpert.csv")

    df = pd.read_csv('total_chexpert.csv',
                     usecols=['Path','No Finding', 'Enlarged Cardiomediastinum',
                              'Cardiomegaly', 'Lung Opacity',
                              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                              'Atelectasis',
                              'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                              'Support Devices'])

    df2 = pd.read_csv('total_chexpert.csv',
                      usecols=['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                               'Lung Opacity',
                               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                               'Atelectasis',
                               'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                               'Support Devices'])

    for idx, x in enumerate(df2.sum(axis=1)):
        if x == 0:
            print("changed index:" + str(idx))
            df.loc[idx, 'No Finding'] = 1

    df.to_csv("total_chexpert.csv")

def mergeMIMIC():
    df = pd.read_csv('train_mimic.csv')
    df2 = pd.read_csv('test_mimic.csv')

    result = pd.concat([df,df2])
    result = result.loc[result['view'] == "frontal"]
    result = result.fillna(0)
    result = result.replace(-1.0,1.0)

    result.to_csv("total_mimic.csv")

    df = pd.read_csv('total_mimic.csv',
                     usecols=['path','No Finding', 'Enlarged Cardiomediastinum',
                              'Cardiomegaly', 'Airspace Opacity',
                              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                              'Atelectasis',
                              'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                              'Support Devices'])

    df2 = pd.read_csv('total_mimic.csv',
                      usecols=['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                               'Airspace Opacity',
                               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                               'Atelectasis',
                               'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                               'Support Devices'])

    for idx, x in enumerate(df2.sum(axis=1)):
        if x == 0:
            print("changed index:" + str(idx))
            df.loc[idx, 'No Finding'] = 1

    df.to_csv("total_mimic.csv")



def mergeTotal():
    dfmimic = pd.read_csv('total_mimic.csv')
    dfchex = pd.read_csv('total_chexpert.csv')

    dfmimic = dfmimic.rename(columns={'Airspace Opacity': 'Lung Opacity','path': 'Path'})
    result = pd.concat([dfmimic, dfchex])

    result.to_csv("mimic_chex.csv")


def stratify():
    labels = ['No Finding', 'Enlarged Cardiomediastinum','Cardiomegaly', 'Lung Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                 'Atelectasis','Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                 'Fracture', 'Support Devices']

    totallabels = ['Path','No Finding', 'Enlarged Cardiomediastinum','Cardiomegaly', 'Lung Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                 'Atelectasis','Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                 'Fracture', 'Support Devices']

    df = pd.read_csv('mimic_chex.csv')
    totalX = df["Path"].values
    totalY = df[labels].values
    totalX = np.expand_dims(totalX, axis=1)

    print("PRE ITERATIVE")
    X_train, y_train, X_test, y_test = iterative_train_test_split(totalX, totalY, 0.2)

    print(X_train.shape)
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

    print("WRITING Train")
    dfTotal2 = pd.DataFrame(columns=totallabels)
    dfTotal2['Path'] = X_train.flatten()
    dfTotal2[labels] = y_train
    dfTotal2.to_csv("train_draft.csv")

    print("WRITING Test")
    dfTotal2 = pd.DataFrame(columns=totallabels)
    dfTotal2['Path'] = X_test.flatten()
    dfTotal2[labels] = y_test
    dfTotal2.to_csv("test.csv")


def stratifyval():
    labels = ['No Finding', 'Enlarged Cardiomediastinum','Cardiomegaly', 'Lung Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                 'Atelectasis','Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                 'Fracture', 'Support Devices']

    totallabels = ['Path','No Finding', 'Enlarged Cardiomediastinum','Cardiomegaly', 'Lung Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                 'Atelectasis','Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                 'Fracture', 'Support Devices']

    df1 = pd.read_csv('train_tmp.csv')
    df2 = pd.read_csv('test_tmp.csv')
    df3 = pd.read_csv('val_tmp.csv')
    df = pd.concat([df1,df2,df3])

    totalX = df["Path"].values
    totalY = df[labels].values
    totalX = np.expand_dims(totalX, axis=1)

    print("PRE ITERATIVE")
    X_train, y_train, X_test, y_test = iterative_train_test_split(totalX, totalY, test_size=0.2)

    X_train, y_train, X_val, y_val = iterative_train_test_split(X_train, y_train, test_size=0.2)

    print("WRITING Train")
    dfTotal2 = pd.DataFrame(columns=totallabels)
    dfTotal2['Path'] = X_test.flatten()
    dfTotal2[labels] = y_test
    dfTotal2.to_csv("test_tmp2.csv")

    print("WRITING Train")
    dfTotal3 = pd.DataFrame(columns=totallabels)
    dfTotal3['Path'] = X_val.flatten()
    dfTotal3[labels] = y_val
    dfTotal3.to_csv("val_tmp2.csv")

    print("WRITING Test")
    dfTotal4 = pd.DataFrame(columns=totallabels)
    dfTotal4['Path'] = X_train.flatten()
    dfTotal4[labels] = y_train
    dfTotal4.to_csv("train_tmp2.csv")


def pathToTemp():
    df_train = pd.read_csv('train.csv')
    df_train['Path'] = '/tmp/test/' + df_train['Path'].astype(str)
    df_test = pd.read_csv('test.csv')
    df_test['Path'] = '/tmp/test/' + df_test['Path'].astype(str)
    df_val = pd.read_csv('val.csv')
    df_val['Path'] = '/tmp/test/' + df_val['Path'].astype(str)

    df_train.to_csv("train_tmp.csv", index=False)
    df_val.to_csv("val_tmp.csv", index=False)
    df_test.to_csv("test_tmp.csv", index=False)

if __name__ == '__main__':
    mergeChexPert()
    mergeMIMIC()
    mergeTotal()
    stratify()
    stratifyval()
    pathToTemp()
    stratifyval()
