
import tensorflow as tf
from PIL import Image
import glob
import numpy as np

X_train = []
y_train = []
X_test = []
y_test = []
for i in range(16):
    for f in glob.iglob("/Users/changmin/Desktop/학교/2018-2/창의설계프로젝트2/train/%d/*" % i):
        X_train.append(np.asarray(Image.open(f).convert('L')))
        y_train.append(np.asarray([i]))

for i in range(16):
    for f in glob.iglob("/Users/changmin/Desktop/학교/2018-2/창의설계프로젝트2/test/%d/*" % i):
        X_test.append(np.asarray(Image.open(f).convert('L')))
        y_test.append(np.asarray([i]))


X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)
X_train = X_train.reshape([len(X_train), 200, 200, 1])
X_test = X_test.reshape([len(X_test), 200, 200, 1])

np.savetxt("X_train.csv", X_train.reshape([-1, 200]), delimiter=',', fmt='%d')
np.savetxt("y_train.csv", y_train.reshape([-1, 1]), delimiter=",", fmt='%d')
np.savetxt("X_test.csv", X_test.reshape([-1, 200]), delimiter=",", fmt='%d')
np.savetxt("y_test.csv", y_test.reshape([-1, 1]), delimiter=",", fmt='%d')

print("data save done")