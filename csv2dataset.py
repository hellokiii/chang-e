import numpy as np
from PIL import Image

X_train = np.loadtxt('X_train.csv', delimiter=',', dtype=np.int32)
X_test = np.loadtxt('X_test.csv', delimiter=',', dtype=np.int32)
y_train = np.loadtxt('y_train.csv', delimiter=',', dtype=np.int32)
y_test = np.loadtxt('y_test.csv', delimiter=',', dtype=np.int32)

X_train = X_train.reshape([-1, 200, 200, 1])
X_test = X_test.reshape([-1, 200, 200, 1])
y_train = y_train.reshape([-1, 1])
y_test = y_test.reshape([-1, 1])

iii = X_test[15].reshape([200, 200])
img = Image.fromarray(iii, 'L')
img.show()
