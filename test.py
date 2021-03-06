import os
import numpy as np
import cv2
from PIL import Image
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.initializers import glorot_normal
from sklearn.model_selection import train_test_split

def load_lego_data():
    # Load lego image data and convert them to RGB matrix
    png_to_rgb = []
    label = []
    size = 50, 50
    for root, dirs, files in os.walk('.'):
        for filenames in files:
            if os.path.splitext(filenames)[1] == '.png':
                image = Image.open('{image}'.format(
                        image=os.path.join(root, filenames))).convert('L')
                image.thumbnail(size, Image.ANTIALIAS)
                png_to_rgb.append(np.array(image).astype('uint8'))
                label.append([int(root[-1])] if root[-2] == '/'
                        else [int(root[-2:])])
    return png_to_rgb, label

def prac_model(num_classes):
    # Create model
    model = Sequential()
    model.add(Conv2D(11, (4, 4), input_shape=(50, 50, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(88, activation='relu'))
    model.add(Dense(33, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    sgd = optimizers.Adam(lr=0.00103)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                    metrics=['accuracy'])
    return model

if __name__ == '__main__':
    X_test, y_test = load_lego_data()
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    X_test = X_test.reshape(X_test.shape[0], 50, 50, 1).astype('float32')
    X_test = X_test / 255
    y_test = np_utils.to_categorical(y_test)

    model = prac_model(num_classes=16)
    model.load_weights('prac8.h5')
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100 - scores[1]*100))

    quit()
