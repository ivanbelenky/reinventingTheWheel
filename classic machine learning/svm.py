import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax, reshape
from numpy.core.numeric import full
import utils as u 
from logit import Logit

plt.style.use("dark_background")

import tensorflow as tf
from tensorflow import keras 


TOLERANCE = 0.001
MAX_ITER = 1E4


class SVM(object):
 
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
        self.data_labels = data_labels
        self.classes = np.unique(self.data_labels)
        self.logits = [Logit(self.batch, self.tolerance, self.max_iter, self.h) 
        for _ in self.classes]
    

    def train(self, data, data_labels):
        self.load(data, data_labels)

        W = []
        for c in self.classes:
            changed_labels = np.copy(data_labels)
            changed_labels = changed_labels.astype('int16')
            changed_labels[changed_labels != c] = -1.
            changed_labels[changed_labels != -1.] = 1.
            amount_of_data = changed_labels[changed_labels == 1].shape[0]
            self.logits[c].train(data, changed_labels)

            print(f"class {c} there are {amount_of_data}")
            W.append(self.logits[c].W)
        self.W = np.vstack(W)

    def predict(self, data_test):
        predicted = []
        for data in data_test:
            probs =  np.array([self.probability(data,c) for c in self.classes])
            predicted.append(probs.argmax())
        predicted = np.array(predicted)

        return predicted
            
    # def probability(self,data, c):
    #     return self.logits[c].probability(data) 

    def probability(self, data, c):
        num = self.logits[c].probability(data)
        denom = np.prod(np.array([self.logits[j].probability(data)
        for j,_ in enumerate(self.logits) if j != c]))
        denom = denom*(1-num)
        return num/denom

def main():

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype('float64')
    x_test = x_test.astype('float64') 

    x_train /= 255
    x_test /= 255

    data = x_train.reshape(x_train.shape[0],-1)
    data_labels = y_train.reshape(-1)
    data_test = x_test.reshape(x_test.shape[0],-1)
    label_test = y_test.reshape(-1)
    
    svm = SVM(batch_size=1)
    svm.train(data, data_labels)
    
    predicted = svm.predict(data_test)
    mask = (predicted == label_test)
    print(mask[mask==True].shape[0]/mask.shape[0])

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    fig, ax = plt.subplots(5,2,figsize=(10,20))
    for i,temp in enumerate(svm.W):
        template = temp.reshape(x_train.shape[1:])
        template += np.abs(np.min(template))
        template = template/np.max(template)*255

        ax[i%5][int(i/5)].imshow(template.astype('uint8'), cmap=plt.get_cmap('gray'))
        ax[i%5][int(i/5)].axis('off')
        ax[i%5][int(i/5)].set_title(f"{i}")

    plt.show()

if __name__ == "__main__":
    main()


        
