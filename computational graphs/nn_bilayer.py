import numpy as np
import tensorflow as tf
from tensorflow import keras

import utils as u


def compute_NN(x_t, y_t, W1, W2):
    x = np.hstack((x_t,1))
    z = W1.dot(x)
    n = np.hstack((u.sgm(z),1))
    y = W2.dot(n)

    y_idx = y_t
    y_t = np.zeros(LAYER2_SIZE)
    y_t[y_idx] = 1
    
    MSE = u.MSE(y,y_t)

    return MSE, y, n, z

def dW_2(MSE, y, y_true, n):
    n = n.reshape(n.shape[0],1)
    grad_mse = (2*(y-y_true)/y_true.shape).reshape(1,-1)
    dW_2 = n.dot(grad_mse)
    return dW_2.T


def dW_1(MSE, y, y_true, W2, z, x_t):
    x = np.hstack((x_t,1))
    grad_mse = (2*(y-y_true)/y_true.shape[0]).reshape(1,-1)

    dMSE_dn = grad_mse.dot(W2)[:,:-1]
    
    dMSE_dn = dMSE_dn.reshape(-1)
    grad = dMSE_dn * u.d_sgm(z)
    grad = grad.reshape(1,grad.shape[0])
    x = x.reshape(x.shape[0],1)
    dW_1 = x.dot(grad)
    return dW_1.T


def MSE(s,y_true):
    return np.mean(np.sum((s-y_true)**2))

def grad_MSE(s,y_true):
    return 2*(s-y_true)/y_true.shape[0]

def L2(W):
    return np.sum(W ** 2)


def grad_L2(W):
    grad = 2 * W
    grad [:,-1] = 0
    return grad


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype(np.float)
x_test = x_test.astype(np.float) 


data = x_train.reshape(x_train.shape[0],-1)
data_labels = y_train.reshape(-1)

data_test = x_test.reshape(x_test.shape[0],-1)
label_test = y_test.reshape(-1)

data -= np.mean(data,axis=0)
data_test -= np.mean(data,axis=0)


LAYER1_SIZE = 50
LAYER2_SIZE = 10
BATCH_SIZE = 50
EPOCHS = 200

N_BATCHS = int(data.shape[0] / BATCH_SIZE)

LAMBD = 1E-2
STEP1 = 1E-5
STEP2 = 1E-5

reg = L2
grad_reg = grad_L2


def train(x_t, y_t):
    W1 = np.random.uniform(-1,1,size=(LAYER1_SIZE, data.shape[1]+1))/1E3
    W2 = np.random.uniform(-1,1,size=(LAYER2_SIZE, LAYER1_SIZE+1))/1E3
    print("\n\nbeginning training...\n\n")
    
    for epoch in range(EPOCHS):
    
        idxs = np.arange(x_t.shape[0])
        np.random.shuffle(idxs)

        for batch in range(N_BATCHS):
            dW1 = 0
            dW2 = 0
            selected_idxs = idxs[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]

            for idx in selected_idxs:
                MSE, y, n, z = compute_NN(x_t[idx], y_t[idx], W1, W2)
                y_true = np.zeros(LAYER2_SIZE)
                y_true[y_t[idx]]= 1
                dW1 += dW_1(MSE, y, y_true, W2, z, x_t[idx])
                dW2 += dW_2(MSE, y, y_true, n)

            W1 = W1-STEP1*(dW1+0.5*LAMBD*grad_reg(W1))
            W2 = W2-STEP2*(dW2+0.5*LAMBD*grad_reg(W2))

        
        print(f"accuracy:",test(data[:1000], data_labels[:1000], W1, W2))
        print(np.mean(dW1),np.mean(dW2))
        print(f"{epoch/EPOCHS*100:.3f}%",end='\r')
        

    return W1, W2

def test(data_test, label_test, W1, W2):
    predicted = []    
    for data, label in zip(data_test, label_test):
        _, y, _, _ = compute_NN(data, label, W1, W2)
        predicted.append(np.argmax(y))
    predicted = np.array(predicted)
    print(np.unique(predicted))
    mask = (predicted == label_test)

    return accuracy(mask)


def accuracy(mask):
    return (mask[mask==True].shape[0]/mask.shape[0])*100



W1, W2 = train(data, data_labels)

print(test(data_test,label_test, W1, W2))



