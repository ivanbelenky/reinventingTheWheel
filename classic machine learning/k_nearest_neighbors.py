""""
from wikipedia.org/wiki/K-nearset_neighors_algorithm
    
    (...) k-NN is a non parametric classification (as the name implies,
    branch of statistics that is not solely base on parametrized families
    of probability distributions). Distribution-free or having specified 
    distribution with parameters unsepcified. Statistical inference is 
    included here. (...) the use of non-parametric methods may be necessary
    when data have a ranking but no clear numerical interpretation, such as 
    when assessing preferences. As they make fewer assumptions, their appli-
    cability is much wider than the corresponding parametric methods. They tend
    to be more robust. 

    Example for non parametric classification are
        Histograms
        Non Parametric regressions
        KNN 
        Support vector machine
    
    Even though they are more robust, this comes at a cost. Robustness implies 
    that these methods are not unduly affected b outliers or other small departures
    from model assumptions.  

    So back to KNN, this algorithm could be a classifier, i.e. the output is a class
    membership. The class most common around its neighbors, is the calss being
    assigned. The output could also be a property value for the object. This value
    is the average of the values of k nearest neighbors. 
"""

import tensorflow as tf
from tensorflow import keras 
import numpy as np
import matplotlib.pyplot as plt
import sys
import utils as u



def k_nearest_neighbors(target, k, data, labels, method='majority', norm='euclidean'):
    if (k < 1) or not isinstance(k, int):
        return None

    function = METHODS[method]
    norm = NORMS[norm]

    return function(target, k, data, labels, norm)

def maj_knn(target, k, data, labels, norm):
    distances = norm(data-target, axis=1)
    sorted_idx = np.argsort(distances)
    k_labels = labels[sorted_idx][:k]
    target_label = np.bincount(k_labels).argmax()

    return target_label


def wght_knn(target, k, data, labels, norm):
    distances = norm(data-target, axis=1)
  
    sorted_idx = np.argsort(distances)
    k_labels = labels[sorted_idx][:k]
    weights = np.sort((1/distances))[::-1][:k]
    
    unique = np.unique(k_labels)
    sum_of_weights = np.array([np.sum(weights[k_labels==u]) for u in unique])
    max_weighted_label = unique[np.argmax(sum_of_weights)]
    
    return max_weighted_label


def timer(func):
    import time
    def time_func(*args):
        now = time.time()
        func(*args)
        print(f"Function Took {time.time()-now} seconds to execute")
    
    return time_func


@timer
def main(database_loader):
    TOTEST = 1000
    K = 20

    (x_train, y_train), (x_test, y_test) = database_loader()
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32') 
    
    data = x_train.reshape(x_train.shape[0],-1)
    data_labels = y_train.reshape(-1)
    print(f"reshaped x_train,y_train data has shape {data.shape, data_labels.shape}")

    x_test = x_test.reshape(x_test.shape[0],-1)
    y_test = y_test.reshape(-1)
    print(f"reshaped x_test,y_test data has shape {x_test.shape, y_test.shape}")
    
    classes = np.unique(data_labels)
    
    idxs = np.array([_ for _ in range(x_test.shape[0])])
    selected_idxs = np.random.choice(idxs, size=TOTEST)
    predicted_classes = []
    real_classes = []
    for idx in selected_idxs:
        target = x_test[idx]
        real_classes.append(classes[y_test[idx]])

        predicted_class = classes[k_nearest_neighbors(
            target, K, data, data_labels, method='majority', norm='euclidean')]
        predicted_classes.append(predicted_class)
    
    predicted_classes = np.array(predicted_classes)
    real_classes = np.array(real_classes)
    
    mask = (predicted_classes == real_classes)
    mask = mask[mask==True]
    accuracy = mask.shape[0]/TOTEST
    print('Accuracy is:', u.accuracy(real_classes, predicted_classes))
    

METHODS = {
    'majority': maj_knn,
    'weighted': wght_knn,
}

NORMS = {
    'euclidean': np.linalg.norm,
}

LOADERS = {
    'cifar': tf.keras.datasets.cifar10.load_data,
    'mnist': tf.keras.datasets.mnist.load_data
}



if __name__ == "__main__":
    main(LOADERS[sys.argv[1]])





