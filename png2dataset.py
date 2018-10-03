import tensorflow as tf
from PIL import Image
import glob
import numpy as np


# np.set_printoptions(threshold=np.nan)
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


X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

X_train = X_train.reshape([len(X_train), 200, 200, 1])
X_test = X_test.reshape([len(X_test), 200, 200, 1])

print(X_train.shape)
print(y_test[15])
print(X_test[1].shape)

iii = X_test[15].reshape([200, 200])
print(y_test[15])
print(y_train[100])
aaa = X_train[100].reshape([200, 200])
img = Image.fromarray(iii, 'L')
img2 = Image.fromarray(aaa, 'L')
img.show()
img2.show()
