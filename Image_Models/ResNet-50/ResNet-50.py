from keras import Model
from keras.applications.resnet50  import ResNet50
from keras_efficientnets.custom_objects import EfficientNetDenseInitializer
from keras.layers import Dense, Activation, GlobalAveragePooling2D, Dropout
import tensorflow as tf
import pandas as pd
from keras_preprocessing import image
from metrics import roc_callback
import numpy as np
import keras.backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from CiclicLR import CyclicLR

K.set_image_data_format('channels_last')
TRAIN = "../../Datasets/MIMIC-CHEXP/train_tmp2.csv"
TEST = "../../Datasets/MIMIC-CHEXP/test_tmp2.csv"
VAL = "../../Datasets/MIMIC-CHEXP/val_tmp2.csv"

BATCH=16
EPOCHS=30
THREAD = 8

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


def createModelGrayscale():
    base_model = ResNet50(weights=None, input_shape=(256,256,1), include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(14, kernel_initializer="he_normal", activation='sigmoid')(x)

    model = Model(base_model.input,x)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', auc_roc])
    model.summary()
    return model


def createModelPretrainedRGB():
    base_model = ResNet50(weights="imagenet", input_shape=(256,256,3), include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(14, kernel_initializer="he_normal", activation='sigmoid')(x)

    model = Model(base_model.input,x)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', auc_roc])
    model.summary()
    return model

def trainGrayscale(model):

    labels = ['No Finding','Enlarged Cardiomediastinum',
              'Cardiomegaly','Lung Opacity','Lung Lesion',
              'Edema','Consolidation','Pneumonia','Atelectasis',
              'Pneumothorax','Pleural Effusion','Pleural Other','Fracture','Support Devices']
    datagen = image.ImageDataGenerator(rescale=1. / 255)

    traindf = pd.read_csv(TRAIN)
    validatedf = pd.read_csv(VAL)
    testdf = pd.read_csv(TEST)


    traingenerator = datagen.flow_from_dataframe(traindf,
                                                 directory=None,
                                                 color_mode='grayscale',
                                                 target_size=(256, 256),
                                                 x_col='Path',
                                                 y_col=labels,
                                                 class_mode="other",
                                                 shuffle=True,
                                                 batch_size=BATCH,
                                                 drop_duplicates=False)

    validategenerator = datagen.flow_from_dataframe(validatedf,
                                                    directory=None,
                                                    color_mode='grayscale',
                                                    target_size=(256, 256),
                                                    x_col='Path',
                                                    y_col=labels,
                                                    class_mode="other",
                                                    shuffle=False,
                                                    batch_size=BATCH,
                                                    drop_duplicates=False)

    testgenerator = datagen.flow_from_dataframe(testdf,
                                                    directory=None,
                                                    color_mode='grayscale',
                                                    target_size=(256, 256),
                                                    x_col='Path',
                                                    y_col=labels,
                                                    class_mode="other",
                                                    shuffle=False,
                                                    batch_size=BATCH,
                                                    drop_duplicates=False)

    print(traingenerator.n)
    print(validategenerator.n)
    print(testgenerator.n)

    filepath = "ResNet-grayscale-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,
                                 mode='min')
    clr = CyclicLR(base_lr=0.0001, max_lr=0.0006, step_size=2000.)
    es = EarlyStopping(monitor="val_loss",mode=min, verbose=1)
    callbacks_list = [checkpoint,clr,es,roc_callback(testgenerator,np.array(testgenerator.labels))]

    model.fit_generator(generator=traingenerator,
                        validation_data=validategenerator,
                        epochs=EPOCHS,
                        steps_per_epoch=traingenerator.n / BATCH,
                        validation_steps=validategenerator.n / BATCH,
                        callbacks=callbacks_list,
                        workers=THREAD,
                        verbose=1)
    model.save_weights("ResNet.h5")
    model.save('ResNet.h5')

def trainRGB(model):

    labels = ['No Finding','Enlarged Cardiomediastinum',
              'Cardiomegaly','Lung Opacity','Lung Lesion',
              'Edema','Consolidation','Pneumonia','Atelectasis',
              'Pneumothorax','Pleural Effusion','Pleural Other','Fracture','Support Devices']
    datagen = image.ImageDataGenerator(rescale=1. / 255)

    traindf = pd.read_csv(TRAIN)
    validatedf = pd.read_csv(VAL)
    testdf = pd.read_csv(TEST)


    traingenerator = datagen.flow_from_dataframe(traindf,
                                                 directory=None,
                                                 color_mode='rgb',
                                                 target_size=(256, 256),
                                                 x_col='Path',
                                                 y_col=labels,
                                                 class_mode="other",
                                                 shuffle=True,
                                                 batch_size=BATCH,
                                                 drop_duplicates=False)

    validategenerator = datagen.flow_from_dataframe(validatedf,
                                                    directory=None,
                                                    color_mode='rgb',
                                                    target_size=(256, 256),
                                                    x_col='Path',
                                                    y_col=labels,
                                                    class_mode="other",
                                                    shuffle=False,
                                                    batch_size=BATCH,
                                                    drop_duplicates=False)

    testgenerator = datagen.flow_from_dataframe(testdf,
                                                    directory=None,
                                                    color_mode='rgb',
                                                    target_size=(256, 256),
                                                    x_col='Path',
                                                    y_col=labels,
                                                    class_mode="other",
                                                    shuffle=False,
                                                    batch_size=BATCH,
                                                    drop_duplicates=False)

    print(traingenerator.n)
    print(validategenerator.n)
    print(testgenerator.n)

    filepath = "ResNet-rgb-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,
                                 mode='min')
    clr = CyclicLR(base_lr=0.0001, max_lr=0.0006, step_size=2000.)
    es = EarlyStopping(monitor="val_loss",mode=min, verbose=1)
    callbacks_list = [checkpoint,clr,es,roc_callback(testgenerator,np.array(testgenerator.labels))]

    model.fit_generator(generator=traingenerator,
                        validation_data=validategenerator,
                        epochs=EPOCHS,
                        steps_per_epoch=traingenerator.n / BATCH,
                        validation_steps=validategenerator.n / BATCH,
                        callbacks=callbacks_list,
                        workers=THREAD,
                        verbose=1)
    model.save_weights("ResNet.h5")
    model.save('ResNet.h5')

def generatePredictionsGrayscale(model):
    labels = ['No Finding', 'Enlarged Cardiomediastinum',
              'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
              'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
              'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    datagen = image.ImageDataGenerator(rescale=1. / 255)
    testdf = pd.read_csv(TEST)
    testgenerator = datagen.flow_from_dataframe(testdf,
                                                    directory=None,
                                                    color_mode='grayscale',
                                                    target_size=(256, 256),
                                                    x_col='Path',
                                                    y_col=labels,
                                                    class_mode="other",
                                                    shuffle=False,
                                                    batch_size=64,
                                                    drop_duplicates=False)
    predictions = model.predict_generator(testgenerator, testgenerator.n / 64, verbose=1)
    array = np.array([predictions, testgenerator.labels])
    np.save("ResNet_GrayScale_Predictions", array)


def generatePredictionsRGB(model):
    labels = ['No Finding', 'Enlarged Cardiomediastinum',
              'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
              'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
              'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    datagen = image.ImageDataGenerator(rescale=1. / 255)
    testdf = pd.read_csv(TEST)
    testgenerator = datagen.flow_from_dataframe(testdf,
                                                    directory=None,
                                                    color_mode='rgb',
                                                    target_size=(256, 256),
                                                    x_col='Path',
                                                    y_col=labels,
                                                    class_mode="other",
                                                    shuffle=False,
                                                    batch_size=64,
                                                    drop_duplicates=False)
    predictions = model.predict_generator(testgenerator, testgenerator.n / 64, verbose=1)
    array = np.array([predictions, testgenerator.labels])
    np.save("ResNet_RGB_Predictions", array)

if __name__ == '__main__':
    #modelgrayscale = createModelGrayscale()
    #trainGrayscale(modelgrayscale)
    #generatePredictionsGrayscale

    modelRGB = createModelPretrainedRGB()
    trainRGB(modelRGB)
    modelRGB.load_weights("filename")
    generatePredictionsRGB(model)

