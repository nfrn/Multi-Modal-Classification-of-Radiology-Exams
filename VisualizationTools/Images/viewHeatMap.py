from dual_path_network import DPN92
from keras_preprocessing.image import img_to_array, load_img
from vis.visualization import visualize_cam
from matplotlib import pyplot as plt
import numpy as np
import time

DPN_LAYER=2369
DPN_WEIGHTS="DPN.hdf5"



VIEW_FILE="./Examples/CXR28_IM-1231-1001.png"

def load_DPN():
    model = DPN92(input_shape=(224, 224, 1),classes=14)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def viewHeatMap(model,layer):
    image = img_to_array(load_img(VIEW_FILE,
                                  color_mode="grayscale", target_size=(224, 224))) / 255.

    image3 = img_to_array(load_img(VIEW_FILE,
                                   color_mode="rgb", target_size=(224, 224))) / 255.

    image2 = np.expand_dims(image, axis=0)

    predicted = model.predict([image2])

    predicted[predicted >= 0.1] = 1
    predicted[predicted < 0.1] = 0
    print(predicted)
    predicted = np.array(predicted)
    indexes = np.where(predicted[0] == 1)[0]
    print(indexes)

    grads = visualize_cam(model, layer, indexes, image)
    plt.imshow(image3)
    plt.imshow(grads, cmap='jet', alpha=0.5)
    plt.show()

    time.sleep(5)



if __name__ == '__main__':
    model = load_DPN()
    model.load_weights(DPN_WEIGHTS)
    print({i: v for i, v in enumerate(model.layers)})
    viewHeatMap(model, DPN_LAYER)