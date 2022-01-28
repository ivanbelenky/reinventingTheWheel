
# Computational graphs 

This package implemented from scratch feedforward neural network framework. In order to achieve this a computational graph module a.k.a. ```cgraph```  was needed in order to represent the neural network. Also this graph module implmented a variety of operation omnipresent in the field. After this another layer of abstraction was needed, the ```Layer``` layer. Finally the FeedForward NeuralNetwork was implmeneted over aboth of this modules in order to give the user easy network generation. 


- [x] Computational Graph  
- [x] Feed Forward Layers
- [x] Feed Forward Networks  

## Usage
Behaviour of other frameworks is replicated in order to build a feedforward neural network.

``` python
from network import FFNN

model = FFNN(
  [100,'relu',40,'tanh',20,'sgm'],
  loss = "CrossEntropy", 
  optimizer = 'BGD', 
  regularizer = 'L2',
  prob = "Softmax", 
  lr = 1E-3
)

model.load(data, labels, test_data, test_labels)
model.train()
```


