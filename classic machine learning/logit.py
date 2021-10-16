from numpy.core.fromnumeric import reshape
from sympy import symbols, diff

import sys, os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("dark_background")

import tensorflow as tf
from tensorflow import keras 

from utils import logit, dlogit


def sim_logit():
    x = symbols('x')
    f = 1/(1+np.e**(-x))
    return f, x


TOLERANCE = 0.0001
MAX_ITER = 1E4




class Logit(object):
    """
    Logistic Regression class. Method of training is BGD.
    Batch size is selectable. Tolerance is setted up regarding
    initialization.
    """
 
    def __init__(self,
        batch_size = 1,
        tolerance = TOLERANCE,
        max_iter = MAX_ITER,
        h = 1E-3
    ):
    
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.batch = batch_size
        self.h = h



    def load(self, data, data_labels):
        self.data = data
        self.data_labels = data_labels.astype('int16')


    def train(self, data, data_labels):
        self.load(data, data_labels)
        shape = data.shape[1]

        W = np.random.random(size=(1,shape))/1E3
        count = 0
        
        while(count < self.max_iter ):
            idxs = np.random.randint(0, shape, size=self.batch)
            labels = self.data_labels[idxs]
            data_points = data[idxs]
            dW = self._dW(W, data_points, labels)
            W = W + dW
            count += 1
            if count%1000 == 0:
                #tol = np.linalg.norm(dW)
                #if tol < self.tolerance:
                #    break
                print(count/self.max_iter, np.mean(dW), end="\r")
          
        self.W = W
    

    def _dW(self, W, data_points, labels):        
        dW = 0
        for data, label in zip(data_points, labels):
            v = data.reshape(data.shape[0],1)
            w = W.dot(v)
            dx_dW = v.T

            if label == 1:
                dW = dW + self.h*dlogit(w)*dx_dW/logit(w)
            elif label == -1:
                dW = dW - self.h*dlogit(w)*dx_dW/(1-logit(w))
            
        dW /= data_points.shape[0]
        return dW

    def criteria(self, f_x):
        if f_x > 0.5:
            return 1
        else:
            return 0

    def predict(self, data_test):
        predicted = []
        for data in data_test:
            v = data.reshape(data.shape[0],1)
            w = self.W.dot(v)
            predicted.append(self.criteria(logit(w))) 
        predicted = np.array(predicted)            

        return predicted

    def probability(self, data_test, label=1):
        v = data_test.reshape(data_test.shape[0], 1)
        w = self.W.dot(v)
        if label == 0:
            return (1-logit(w))
        return logit(w)

def main():

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32') 
    x_train /= 255
    x_test /= 255

    data = x_train.reshape(x_train.shape[0],-1)
    data_labels = y_train.reshape(-1)
    data_test = x_test.reshape(x_test.shape[0],-1)
    label_test = y_test.reshape(-1)

    train_idx = (data_labels == 1) | (data_labels == 0)
    test_idx = (label_test == 1) | (label_test == 0)

    changed_labels = np.copy(data_labels)
    changed_labels = changed_labels.astype('int16')
    changed_labels[changed_labels == 0] = -1

    logreg = Logit(batch_size=10)
    logreg.train(data[train_idx], changed_labels[train_idx])
    
    predicted = logreg.predict(data_test[test_idx])
    mask = (predicted == label_test[test_idx])

    print(mask[mask==True].shape[0]/mask.shape[0])

    template = logreg.W.reshape(x_train.shape[1:])
    template += np.abs(np.min(template))
    template = template/np.max(template)*255
    plt.imshow(template.astype('uint8'))
    plt.axis('off')
    plt.show()


def learning_rate_dependence():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32') 
    x_train /= 255
    x_test /= 255

    data = x_train.reshape(x_train.shape[0],-1)
    data_labels = y_train.reshape(-1)
    data_test = x_test.reshape(x_test.shape[0],-1)
    label_test = y_test.reshape(-1)

    train_idx = (data_labels == 1) | (data_labels == 0)
    test_idx = (label_test == 1) | (label_test == 0)
    
    learning_rate = np.linspace(1E-5,1E-1, num=30)
    batch_sizes = [5,10,20,40,60,100,400]
    
    accuracies = []
    for j, batch in enumerate(batch_sizes):
        subacc = []
        for i,rate in enumerate(learning_rate):
            logreg = Logit(batch_size=batch, h=rate)
            logreg.train(data[train_idx], data_labels[train_idx])
            
            predicted = logreg.predict(data_test[test_idx])
            mask = (predicted == label_test[test_idx])
            subacc.append(mask[mask==True].shape[0]/mask.shape[0])
        print(f"{j} out of {len(batch_sizes)}")
        accuracies.append(subacc)

    print(accuracies)
    
    fig, ax = plt.subplots()
    for batch,acc in zip(batch_sizes,accuracies):
        ax.plot(learning_rate, acc, label=f"batchSize = {batch}" )

    ax.legend(loc=2)    
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Accuracy")
    ax.set_title("BGD: Batch Size vs Accuracy. Fixed #iterations=1E4")
    ax.set_xscale('log')
    plt.show()


if __name__== "__main__":
    main()
    #learning_rate_dependence()