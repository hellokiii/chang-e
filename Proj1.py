import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
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
                #image = Image.open('{image}'.format(image=os.path.join(root, filenames)))
                image = Image.open('{image}'.format(
                        image=os.path.join(root, filenames))).convert('L')
                image.thumbnail(size, Image.ANTIALIAS)
                #png_to_rgb.append(np.array(image)[:, :, :3].astype('uint8'))
                png_to_rgb.append(np.array(image).astype('uint8'))
                label.append([int(root[-1])] if root[-2] == '/'
                        else [int(root[-2:])])
    return png_to_rgb, label

def prac_model(num_classes):
    # Create model
    model = Sequential()
    #model.add(Conv2D(11, (4, 4), input_shape=(50, 50, 3), activation='relu'))
    model.add(Conv2D(11, (4, 4), input_shape=(50, 50, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    """
    model.add(Conv2D(20, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    """
    model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(Dense(88, activation='relu'))
    model.add(Dense(33, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    sgd = optimizers.Adam(lr=0.000097)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                    metrics=['accuracy'])

    return model

if __name__ == '__main__':
    X_original, y_original = load_lego_data()
    X_train, X_test, y_train, y_test = train_test_split(X_original, y_original,
                                        test_size=0.2, random_state=41)
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    # Reshape to be [# of samples][width][height][pixels]
    #X_train = X_train.reshape(X_train.shape[0], 50, 50, 3).astype('float32')
    #X_test = X_test.reshape(X_test.shape[0], 50, 50, 3).astype('float32')
    X_train = X_train.reshape(X_train.shape[0], 50, 50, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 50, 50, 1).astype('float32')
    # Normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255
    # One-hot encoded outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    #num_classes = y_test.shape[1]

    # Build the model
    model = prac_model(num_classes=16)
    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
                epochs=150, batch_size=50, verbose=2)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100 - scores[1]*100))

    model.save('prac7.h5')

    """
    weight_matrix = []
    for layer in model.layers:
        weight_matrix.append(layer.get_weights())
    print(weight_matrix)
    """

    """
    test_image_index = 15
    #prediction = model.predict(X_test[test_image_index].reshape(1, 50, 50, 3))
    prediction = model.predict(X_test[test_image_index].reshape(1, 50, 50, 1))
    label = y_test[test_image_index]
    prediction = np.argmax(prediction, axis=1)

    # Print X_test[test_image_index].shape
    plt.title('Label is {label}, Prediction is {pred}'.format(label=label,
            pred=prediction))
    #plt.imshow(X_test[test_image_index].reshape(50, 50, 3), cmap='gray')
    plt.imshow(X_test[test_image_index].reshape(50, 50, 1))
    plt.show()
    """

    quit()
