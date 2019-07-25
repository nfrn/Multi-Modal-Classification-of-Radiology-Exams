from keras import Model
from keras_efficientnets import EfficientNetB3
from keras_efficientnets.custom_objects import EfficientNetDenseInitializer
from keras.layers import Dense,Activation
import tensorflow as tf
import pandas as pd
from keras_preprocessing import image
from metrics import roc_callback
import numpy as np
import keras.backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from CiclicLR import CyclicLR

K.set_image_data_format('channels_last')
TRAIN = "../../Datasets/MIMIC-CHEXP/train.csv"
TEST = "../../Datasets/MIMIC-CHEXP/test.csv"
VAL = "../../Datasets/MIMIC-CHEXP/val.csv"

BATCH=32
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


def createModel():
    net = EfficientNetB3((256,256,3),weights='imagenet',include_top=False,pooling='avg')
    x = Dense(14, kernel_initializer=EfficientNetDenseInitializer())(net.output)
    x = Activation('sigmoid')(x)
    model = Model(net.input,x)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', auc_roc])
    model.summary()
    return model

def train(model):

    labels = ['No Finding','Enlarged Cardiomediastinum',
              'Cardiomegaly','Lung Opacity','Lung Lesion',
              'Edema','Consolidation','Pneumonia','Atelectasis',
              'Pneumothorax','Pleural Effusion','Pleural Other','Fracture','Support Devices']
    datagen = image.ImageDataGenerator(rotation_range=5,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.01,
                                       zoom_range=[0.05, 0.1], horizontal_flip=True,
                                       vertical_flip=False, fill_mode='reflect',
                                       brightness_range=[0.1, 0.2],
                                       data_format="channels_last")

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

    datagen = image.ImageDataGenerator()
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

    filepath = "EfiNet-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,
                                 mode='min')
    clr = CyclicLR(base_lr=0.0001, max_lr=0.0006, step_size=2000.)
    callbacks_list = [checkpoint,clr,
                      roc_callback(testgenerator,np.array(testgenerator.labels))]

    model.fit_generator(generator=traingenerator,
                        validation_data=validategenerator,
                        epochs=EPOCHS,
                        steps_per_epoch=traingenerator.n / BATCH,
                        validation_steps=validategenerator.n / BATCH,
                        callbacks=callbacks_list,
                        workers=THREAD,
                        verbose=1)

    model.save_weights("EfiNet.h5")
    model.save('EfiNet.h5')

if __name__ == '__main__':
    model = createModel()
    train(model)

