from collections import Counter

import numpy as np
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix

PATH = "NOTEEVENTS.csv"
PATH_RADIO = "RADIO_1.csv"
PATH_RADIO2 = "RADIO_2.csv"
PATH_RADIO3 = "RADIO_3.csv"
PATH_RADIO4 = "RADIO_4.csv"
PATH_RADIO5 = "RADIO_5.csv"
PATH_RADIO6 = "RADIO_6.csv"
PATH_RADIO7 = "RADIO_7.csv"
PATH_ICD = "DIAGNOSES_ICD.csv"
PATH_ICD2 = "D_ICD_DIAGNOSES.csv"
ITEMS= {'Enlarged Cardiomediastinum':['1642','1643','1648','1649','1971','2125','5193'],
        'Cardiomegaly':['4293'],
        'Airspace Opacity':['79319'],   #https://www.aapc.com/memberarea/forums/27281-diagnosis-opacity-finding-chest-xray.html
        'Lung Lesion':['51889','5172','5178','74860','74869','86120','86121','86122','86130','86131','86132','9471','79319'],
        'Edema':['5061','5184','7823'],
        'Consolidation':["5078",'51889','79319','486','481'], #Consolidation must be present to diagnose https://www.aapc.com/memberarea/forums/37920-lung-consolidation.html
        'Pneumonia':['01166','00322','01160','01161','01162','01163','01164','01165',
                     '0413','0551','0382','11505','11515','0730','48249','48281','48282',
                     '48283','4800','4801','4802','4803','4808','4809','481','4820','4821',
                     '4822','48230','48231','48232','48239','48240','48241','48242','48284','48289',
                     '4829','4830','4831','4838','4841','4843','4845',
                     '4846','4847','4848','485','486','4870','48801',
                     '48811','48881','51630','51630','51635','51636',
                     '51637','5171','7700','V066','99731','99732',
                     'V0382','V1261'],
        'Atelectasis':['7704','7705'],
        'Pneumothorax':['01170','01171','01172','01173',
                        '01174','01175','01176','5120',
                        '5121','51281','51282','51283',
                        '51289','8600','8601'],
        'Pleural Effusion':['51181','5119','5111'],
        'Pleural Other':["5110"],
        'Fracture':['80700','80710','8190','8191','V5427',
                    '80701','80702','80703','80704','80705',
                    '80706','80707','80708','80709','81103',
                    '81109','81110','80711','80712','80713',
                    '80714','80715','80716','80717','80718',
                    '80719','8072','8073','8074','81000',
                    '81001','81002','81003','81010','81011',
                    '81012','81013','81100','81101','81102',
                    '81103','73311','81111','81112','81113',
                    '81119','80500','80501','80502','80503',
                    '80504','80505','80506','80507','80508',
                    '80510','80511','80512','80513','80514',
                    '80515','80516','80517','80518','8052',
                    '8053','8054','8055','8058','8059','80600',
                    '80601','80602','80603','80604','80605',
                    '80606','80607','80608','80609','80610',
                    '80611','80612','80613','80614','80615',
                    '80616','80617','80618','80619','80620',
                    '80621','80622','80623','80624','806025',
                    '80626','80627','80628','80629','80630',
                    '80631','80632','80633','80634','80635',
                    '80636','80637','80638','80639','8064',
                    '8065','8068','8069'],
        'Support Devices':['V4321','V4500','V4509','99600','99609','9961','9962','99661','V5339','99672','99674']}



def step1():
    df = pd.read_csv(PATH)
    df_radiology = df.loc[df['CATEGORY'] == 'Radiology']
    df_radiology.to_csv(PATH_RADIO, columns=['TEXT',"SUBJECT_ID","HADM_ID"])


def step2():
    df = pd.read_csv(PATH_RADIO)
    df2 = pd.read_csv(PATH_ICD)
    df3 = pd.read_csv(PATH_ICD2)
    dftotal = pd.merge(df,df2, on=["SUBJECT_ID", "HADM_ID"])
    print(dftotal.head(5))
    dftotal2 = pd.merge(dftotal, df3, on="ICD9_CODE")
    print(dftotal2.head(5))
    dftotal2.to_csv(PATH_RADIO2, columns=['TEXT', "SUBJECT_ID", "HADM_ID", "ICD9_CODE",
                                          "SHORT_TITLE", "LONG_TITLE"])

def step3():
    df = pd.read_csv(PATH_RADIO2)
    print(len(df.index))
    for label in ITEMS.keys():
        df[label] = 0
    for label in ITEMS.keys():
        print(label)
        for code in ITEMS.get(label):
            df.loc[df['ICD9_CODE'] == code,label]=1
    print(df.head(5))
    df.to_csv(PATH_RADIO3)

def test1():
    df = pd.read_csv(PATH_RADIO2)
    print(df.iloc[:,1:])


def step4():
    df = pd.read_csv(PATH_RADIO, usecols=['TEXT','SUBJECT_ID', 'HADM_ID'])
    df2 = pd.read_csv(PATH_ICD, usecols=['SUBJECT_ID', 'HADM_ID','SEQ_NUM','ICD9_CODE'])
    dftotal = pd.merge(df,df2, on=["SUBJECT_ID", "HADM_ID"])
    print(dftotal.head(15))
    dftotal.to_csv(PATH_RADIO2, columns=['TEXT', "SUBJECT_ID", "HADM_ID",'SEQ_NUM',"ICD9_CODE"])

def group():
    df = pd.read_csv(PATH_RADIO3,usecols=['TEXT','SUBJECT_ID', 'HADM_ID','Enlarged Cardiomediastinum',
                                          'Cardiomegaly','Airspace Opacity','Lung Lesion','Edema',
                                          'Consolidation','Pneumonia','Atelectasis','Pneumothorax',
                                          'Pleural Effusion','Pleural Other','Fracture','Support '
                                                                                        'Devices'])
    df = df.groupby(['TEXT',"SUBJECT_ID", "HADM_ID"]).sum()
    print(df.head(15))
    df.to_csv(PATH_RADIO4)


def reorder():
    df = pd.read_csv(PATH_RADIO4,
                     usecols=['TEXT', 'Enlarged Cardiomediastinum','Cardiomegaly', 'Airspace Opacity',
                              'Lung Lesion', 'Edema','Consolidation', 'Pneumonia', 'Atelectasis',
                              'Pneumothorax','Pleural Effusion', 'Pleural Other', 'Fracture',
                              'Support Devices'])

    df["No Finding"] = 0

    newOrder = ['TEXT','No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Airspace Opacity','Lung Lesion',
                'Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax','Pleural Effusion',
                'Pleural Other','Fracture','Support Devices']

    df = df.reindex(columns=newOrder)
    df.to_csv(PATH_RADIO5)


def stratify():
    df = pd.read_csv(PATH_RADIO6, usecols=['TEXT', 'No Finding', 'Enlarged Cardiomediastinum',
                                           'Cardiomegaly', 'Airspace Opacity','Lung Lesion',
                                           'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                                           'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                                           'Fracture', 'Support Devices'], engine='python' )
    totalX = df["TEXT"].values
    totalY = df[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis','Pneumothorax',
                 'Pleural Effusion', 'Pleural Other','Fracture', 'Support Devices']].values

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

    print("WRITING Train")

    dfTotal2 = pd.DataFrame(columns=["TEXT", 'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis','Pneumothorax',
                 'Pleural Effusion', 'Pleural Other','Fracture', 'Support Devices'])
    dfTotal2['TEXT'] = X_train.flatten()
    dfTotal2[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis','Pneumothorax',
                 'Pleural Effusion', 'Pleural Other','Fracture', 'Support Devices']] = y_train
    dfTotal2.to_csv("train.csv")

    print("WRITING Test")

    dfTotal2 = pd.DataFrame(columns=["TEXT", 'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis','Pneumothorax',
                 'Pleural Effusion', 'Pleural Other','Fracture', 'Support Devices'])
    dfTotal2['TEXT'] = X_test.flatten()
    dfTotal2[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis','Pneumothorax',
                 'Pleural Effusion', 'Pleural Other','Fracture', 'Support Devices']] = y_test
    dfTotal2.to_csv("test.csv")



def binarize():
    df = pd.read_csv(PATH_RADIO5)
    cols = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Airspace Opacity',
                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis','Pneumothorax',
                 'Pleural Effusion', 'Pleural Other','Fracture', 'Support Devices']
    for label in cols:
        print(label)
        df[label] = (df[label]>1).astype(int)
        print(df[label].head(100))
    df.to_csv(PATH_RADIO6)

if __name__ == '__main__':
    step1()
    step2()
    step3()
    group()
    reorder()
    binarize()
    stratify()