from layers import ActivationLayer, WeightLayer, ProbabilityLayer, LossLayer, InputLayer
import operations as op
import cgraph as cg

import numpy as np

LEARNINGRATE = 1E-3
LAMBD = 1E-2
EPOCHS = 200
BATCHSIZE = 10


class FFNN(object):

    def __init__(
        self,
        layers, 
        loss = "CrossEntropy", 
        optimizer = 'BGD', 
        regularizer = 'L2',
        prob = "Softmax", 
        lr = LEARNINGRATE, 
        lambd = LAMBD,
        batch_size = BATCHSIZE,
        epochs = EPOCHS):

        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.lr = lr
        self.lambd = lambd
        self.batch_size = batch_size
        self.epochs = epochs
        self.prob = prob


    def load(self, data, data_labels, test_data, test_labels):
        self.data = np.hstack([data,np.ones([data.shape[0],1])])
        self.labels = data_labels
        self.test_data = np.hstack([test_data,np.ones((test_data.shape[0],1))])
        self.test_labels = test_labels
        
        self.classes = np.unique(data_labels)
        self.n_batchs  = int(self.data.shape[0] /self.batch_size)

        self.build_graph()


    def build_graph(self):

        self.net = []
        self.nodes = []

        out_shape = self._build_weight_activation_layers()
        self._build_last_layer_to_prob(out_shape)
        self._build_loss_layer()
        
        self._add_regularizers()

        for layer in self.net:
            self.nodes.extend(layer.nodes)  
        self.nodes.extend(self.regularizers)

        self.graph = cg.cGraph(self.nodes)


    def train(self):     
        self.train_data = []

        self.graph.compute()
        
        gradients_list = [self.graph.gradient(self.loss_node, W) for W in self.all_weights]
        reg_grad_list = [self.graph.gradient(reg, W) for reg,W in zip(self.regularizers, self.all_weights)]

        dreg_dW = np.zeros(len(reg_grad_list))

        
        for epoch in range(EPOCHS):
            
            idxs = np.arange(self.data.shape[0])
            np.random.shuffle(idxs)

            for batch in range(self.batch_size):
                selected_idxs = idxs[batch * self.batch_size : (batch + 1) * self.batch_size]

                for i,idx in enumerate(selected_idxs):
                    self.input.value = self.data[idx].T
                    self.true_label.value[self.labels[idx]] = 1 
                    self.graph.compute()
                    
                    gradients_list = [self.graph.gradient(self.loss_node, W) for W in self.all_weights]
                    reg_grad_list = [self.graph.gradient(reg, W) for reg,W in zip(self.regularizers, self.all_weights)]


                    self.true_label.value[self.labels[idx]] = 0
                    if i == 0:
                        dW = [self.graph.propagate_gradient(gl) for gl in gradients_list]
                
                    dW = [dW[i]+self.graph.propagate_gradient(gl) for i,gl in enumerate(gradients_list)]


                dreg_dW = [self.graph.propagate_gradient(gl) for gl in reg_grad_list]
            
                #print(batch/self.batch_size)

            for i,W in enumerate(self.all_weights):
                W.value -= self.lr*(dW[i] + self.lambd*dreg_dW[i])

            self.train_data.append([epoch, self.test()])
            print(f"Accuracy:{self.train_data[-1][1]:.2f} %")

    def test(self):
        predicted = []
        for data, label in zip(self.test_data, self.test_labels):
            self.input.value = data.T
            self.graph.compute()
            predicted.append(np.argmax(self.prediction.value.reshape(-1)))
        
        predicted = np.array(predicted)
        mask = (predicted == self.test_labels)
    
        return self.accuracy(mask)


    def accuracy(self, mask):
        return (mask[mask==True].shape[0]/mask.shape[0])*100

        




    def _build_weight_activation_layers(self):
        # Weight and activation layers   
        #first input
        
        self.net.append(InputLayer(cg.Input(self.data[0].reshape(self.data[0].shape[0],1))))        
        self.input = self.net[0].out
        input_shape = self.input.value.shape[0]
        self.net.append(WeightLayer(self.net[0].out,self.layers[0],input_shape))
        input_shape = self.layers[0]

        
        for i,layer in enumerate(self.layers[1:]):
            
            if isinstance(layer, int):
                self.net.append(WeightLayer(self.net[i+1].out, layer, input_shape))
                input_shape = layer

            elif isinstance(layer,str):
                self.net.append(ActivationLayer(self.net[i+1].out, layer))

            else:
                raise Exception("Wrong layer type")

        return input_shape
    
    def _build_last_layer_to_prob(self, input_shape):
        # last layer to prob
        to_prob = self.net[-1].out
        prob_shape = self.classes.shape[0]
        self.net.append(ProbabilityLayer(to_prob, prob_shape, input_shape, prob = self.prob))
        self.prediction = self.net[-1].out


    def _build_loss_layer(self):
        true_label = np.zeros(self.classes.shape[0])
        true_label[self.labels[0]] = 1
        self.true_label = cg.Input(value = true_label)
        self.net.append(LossLayer(self.net[-1].out, 1, self.true_label, loss=self.loss))
        self.loss_node = self.net[-1].out


    def _add_regularizers(self):
        REGULARIZERS = {"L2":op.L2, "L1":op.L1}
        self.all_weights = [layer.weights for layer in self.net if isinstance(layer, (WeightLayer, ProbabilityLayer))]
        self.regularizers = [REGULARIZERS[self.regularizer]([weight]) for weight in self.all_weights]
        
