
# Larger CNN for the MNIST Dataset
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt
import numpy as np
from keras import backend as K
from PIL import Image
import glob


X_train = []
y_train = []
X_test = []
y_test = []

for i in range(16):
	for index, f in enumerate(glob.iglob("/Users/changmin/Desktop/학교/2018-2/창의설계프로젝트2/train/%d/*" % i)):
		if index % 5 == 0:
			X_test.append(np.asarray(Image.open(f).convert('L')))
			y_test.append(np.asarray([i]))
		else:
			X_train.append(np.asarray(Image.open(f).convert('L')))
			y_train.append(np.asarray([i]))

# for i in range(16):
#     for f in glob.iglob("/Users/changmin/Desktop/학교/2018-2/창의설계프로젝트2/test/%d/*" % i):
#         X_test.append(np.asarray(Image.open(f).convert('L')))
#         y_test.append(np.asarray([i]))
#

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

X_train = X_train.reshape([len(X_train), 200, 200, 1])
X_test = X_test.reshape([len(X_test), 200, 200, 1])

print(X_train.shape)
print(y_train.shape)
# reshape to be [samples][width][height][pixels]
# X_train = X_train.reshape(X_train.shape[0], 32, 32, 3).astype('float32')
# X_test = X_test.reshape(X_test.shape[0], 32, 32, 3).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# define the larger model
def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(15, (5, 5), input_shape=(200, 200, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(30, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten()) # for fully connected
	model.add(Dense(128, activation='relu'))
	xavier = keras.initializers.glorot_normal()
	model.add(Dense(50, activation='relu', kernel_initializer=xavier))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = larger_model()


# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

num = 15
prediction = model.predict(X_test[num].reshape(1,200,200,1))
label = y_test[num]
prediction = np.argmax(prediction, axis=1)


print('Label is {label}, Prediction is {pred}'.format(label=label, pred=prediction))
img = Image.fromarray(X_test[num].reshape([200, 200]), 'L')
img.show()

quit()

