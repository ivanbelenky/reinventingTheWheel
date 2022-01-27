import sys
sys.path.append(sys.path[0]+'/../')


from network import FFNN

import numpy as np
import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype(np.float)
x_test = x_test.astype(np.float) 

data = x_train.reshape(x_train.shape[0],-1)
data_labels = y_train.reshape(-1)

data_test = x_test.reshape(x_test.shape[0],-1)
label_test = y_test.reshape(-1)

data -= np.mean(data,axis=0)
data_test -= np.mean(data,axis=0)

model = FFNN([100,"tanh"], lr=1E-3, batch_size = 50,lambd=1E-6)
model.load(data, data_labels, data_test, label_test)
model.train()

