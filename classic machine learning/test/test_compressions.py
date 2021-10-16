import sys
sys.path.append(sys.path[0]+'/../')

import utils as u

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.type_check import real
plt.style.use("dark_background")

import tensorflow as tf
from tensorflow import keras 


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32') 

data = x_train.reshape(x_train.shape[0],-1)
data_labels = y_train.reshape(-1)

L = int(data.shape[1]*0.8)
D = u.LDA_compression(data[:5000], data_labels[:5000], l=L)
#D = u.PCA_compression(data[:5000],l=L)


