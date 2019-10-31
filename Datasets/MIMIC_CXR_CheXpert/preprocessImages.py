import pandas as pd
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import save_img
import os.path
import errno

def resize():
    traindf = pd.read_csv("train.csv")
    testdf = pd.read_csv("test.csv")
    valdf = pd.read_csv("val.csv")
    dfTotal = pd.concat([traindf, testdf, valdf])
    total = len(dfTotal.index)
    counter = 0
    for path in dfTotal["Path"]:
        counter+=1
        print((counter/total)*100)
        filename = "./"+ path
        if os.path.exists(os.path.dirname(filename)):
            array = img_to_array(load_img(filename, target_size=(256, 256)))
            filename2 = "./resized/"+ path
            if not os.path.exists(os.path.dirname(filename2)):
                try:
                    os.makedirs(os.path.dirname(filename2), exist_ok = True)
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        print("race condition")
                        raise
            save_img(filename2,array)
        else:
            print(filename)


def check(filename):
    df = pd.read_csv(filename)
    for idx, path in enumerate(df["Path"]):
        filename2 = "./resized/" + path
        filename = "./" + path
        print(idx)
        if not os.path.isfile(filename2):
            if os.path.isfile(filename):
                array = img_to_array(load_img(filename, target_size=(256, 256)))
                if not os.path.exists(os.path.dirname(filename2)):
                    try:
                        os.makedirs(os.path.dirname(filename2), exist_ok=True)
                    except OSError as exc:  # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            print("race condition")
                            raise
                save_img(filename2, array)
            else:
                print(filename)
                df = df.drop(df.index[idx])
    newfilename = "checked_"+filename
    df.to_csv(newfilename)

def check2():
    df = pd.read_csv("train.csv")
    for idx, path in enumerate(df["Path"]):
        print(idx)
        filename = "./resized/" + path
        if not os.path.isfile(filename):
            print(filename)
            df = df.drop(df.index[idx])
    df['Path'] = '/tmp/resized/' + df['Path'].astype(str)
    df.to_csv("train_tmp.csv")

    df = pd.read_csv("val.csv")
    for idx, path in enumerate(df["Path"]):
        print(idx)
        filename = "./resized/" + path
        if not os.path.isfile(filename):
            print(filename)
            df = df.drop(df.index[idx])
    df['Path'] = '/tmp/resized/' + df['Path'].astype(str)
    df.to_csv("val_tmp.csv")

    df = pd.read_csv("test.csv")
    for idx, path in enumerate(df["Path"]):
        print(idx)
        filename = "./resized/" + path
        if not os.path.isfile(filename):
            print(filename)
            df = df.drop(df.index[idx])
    df['Path'] = '/tmp/resized/' + df['Path'].astype(str)
    df.to_csv("test_tmp.csv")

if __name__ == '__main__':
    resize()
    check("train.csv")
    check("val.csv")
    check("test.csv")
    check2()