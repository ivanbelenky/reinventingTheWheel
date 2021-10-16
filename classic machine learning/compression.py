import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.type_check import real
plt.style.use("dark_background")

import tensorflow as tf
from tensorflow import keras 

import utils as u
from k_nearest_neighbors import k_nearest_neighbors

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32') 

data = x_train.reshape(x_train.shape[0],-1)[:2000]
data_labels = y_train.reshape(-1)[:2000]

x_test = x_test.reshape(x_test.shape[0],-1)
y_test = y_test.reshape(-1)


N=5
L = int(data.shape[1]*0.5)
fig, ax = plt.subplots(N,2,figsize=(10,10))
D = u.PCA_compression(data, l=L)
#D = u.LDA_compression(data, data_labels, l=L)

for i in range(N):
    x = x_test[np.random.randint(0,x_test.shape[0])]
    new_x = D.T.dot(D).dot(x)
    old_image = x.reshape(28,28)
    new_image = new_x.reshape(28,28)
    
    ax[i][0].imshow(old_image.astype('uint8'), cmap=plt.get_cmap('gray'))
    ax[i][1].imshow(new_image.astype('uint8'), cmap=plt.get_cmap('gray'))
    ax[i][0].axis('off')
    ax[i][1].axis('off')
plt.show()


NUMCOMPRESSIONS = 30
TOTEST = 100
K = 5

classes = np.unique(data_labels)

accuracy_pca = []
els = []

for i in np.linspace(0.02,1,num=NUMCOMPRESSIONS):
    L = int(data.shape[1]*i)
    D = u.PCA_compression(data, l=L)
    #D = u.LDA_compression(data, data_labels, l=L)   

    new_data = D.dot(data.T).T
    new_test = D.dot(x_test.T).T

    idxs = np.array([_ for _ in range(x_test.shape[0])])
    selected_idxs = np.random.choice(idxs, size=TOTEST)
    predicted_classes = []
    real_classes = []
    for idx in selected_idxs:
        target = new_test[idx]
        real_classes.append(classes[y_test[idx]])

        predicted_class = classes[k_nearest_neighbors(
            target, K, new_data, data_labels, method='majority', norm='euclidean')]
        predicted_classes.append(predicted_class)

    predicted_classes = np.array(predicted_classes)
    real_classes = np.array(real_classes)

    accuracy_pca.append(u.accuracy(predicted_classes, real_classes))
    els.append(L)

    print(i*100, "%")

fig,ax = plt.subplots(figsize=(10,10))
ax.plot(els,accuracy_pca, label="PCA")
ax.set_title("Accuracy vs Components Used")
ax.legend(loc=2)

plt.show()
