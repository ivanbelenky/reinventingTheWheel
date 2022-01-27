from cgraph import UnaryOperation, BinaryOperation

import utils as u
import numpy as np



###########################################################
#                       Unary                            #
###########################################################


class Tanh(UnaryOperation):
    def __init__(self, inputs) -> None:
        super().__init__(inputs=inputs)
        self.symbol = "tanh"

    def compute(self):
        self.value = np.tanh(self.input.value)
        return self.value

    def _gradient(self):       
        gradient = np.diag((1-(np.tanh(self.value))**2).reshape(-1))
        self.gradient = gradient
        return self.gradient

class Sigmoid(UnaryOperation):
    def __init__(self, inputs) -> None:
        super().__init__(inputs=inputs)
        self.symbol = "sgm"


    def compute(self):
        self.value =  u.sgm(self.input.value)
        return self.value

    def _gradient(self):       
        gradient = np.diag((u.d_sgm(self.input.value)).reshape(-1))
        self.gradient = gradient
        return self.gradient


class L1(UnaryOperation):
    def __init__(self, inputs) -> None:
        super().__init__(inputs)
        self.symbol = "L1"
        

    def compute(self):
        self.value = np.linalg.norm(self.input.value,1)
        return self.value 
    
    def _gradient(self):
        gradient = np.zeros(shape=self.input.value.shape)
        _L1 = self.compute()
        gradient = self.input.value/_L1
        self.gradient = gradient.T
        return self.gradient


class L2(UnaryOperation):
    def __init__(self, inputs) -> None:
        super().__init__(inputs)
        self.symbol = "L2"


    def compute(self):
        self.value = np.linalg.norm(self.input.value)
        return self.value
        
    def _gradient(self):
        gradient = np.zeros(shape=self.input.value.shape)
        _L2 = self.compute()
        gradient = self.input.value/_L2
        self.gradient = gradient.T
        return self.gradient


class ReLU(UnaryOperation):
    def __init__(self, inputs) -> None:
        super().__init__(inputs=inputs)
        self.symbol = "ReLU"


    def compute(self):
        self.value = u.relu(self.input.value)
        return self.value
        

    def _gradient(self):       
        gradient = np.diag(u.d_relu(self.value))
        self.gradient = gradient
        return self.gradient


class Softmax(UnaryOperation):
    def __init__(self, inputs) -> None:
        super().__init__(inputs=inputs)
        self.symbol = "Softmax"
    
    def compute(self):
        z_max = np.max(self.input.value)
        denom = np.sum(np.exp(self.input.value-z_max))
        self.value = np.exp(self.input.value-z_max)/denom
        return self.value

    def _gradient(self):
        z_max = np.max(self.input.value)
        denom = np.sum(np.exp(self.input.value-z_max))
        exp = np.exp(self.input.value-z_max)
        gradient = np.diag(exp/denom) - exp.dot(exp.T)/denom**2

        self.gradient = gradient
        return self.gradient



###########################################################
#                       Binary                            #
###########################################################



class Sum(BinaryOperation):
    def __init__(self, inputs) -> None:
        super().__init__(inputs=inputs)
        self.symbol = "+"
        self.value = None
        self.gradient = [None, None]

    def compute(self):
        self.value = self.inputs[0].value+self.inputs[1].value
        return  self.value

    def _gradient(self):
        shape = self.inputs[0].value.shape
        self.gradient[0] = np.identity(shape[0])
        self.gradient[1] = np.identity(shape[0])

class Wx(BinaryOperation):
    """this will be just matrix times a vector because, will do efficiently the tensor product """
    def __init__(self, inputs) -> None:
        super().__init__(inputs)
        self.symbol = "."
        self.value = None
        self.gradient = [None, None]

    def compute(self):
        self.value = self.inputs[0].value.dot(self.inputs[1].value)
        return self.value

    def _gradient(self):
        self.gradient[0] = self.inputs[1].value.reshape(1,-1)
        self.gradient[1] = self.inputs[0].value
        return self.gradient


class TensorDot(BinaryOperation):
    def __init__(self, inputs) -> None:
        super().__init__(inputs=inputs)
        self.symbol = "*"
        self.value = None


    def compute(self):
        W = self.inputs[0].value
        x = self.inputs[1].value
        return np.tensordot(W,x,axes=1) 


    def _gradient(self):
        shapeW = np.array(self.inputs[0].value.shape)
        shapex = np.array(self.inputs[1].value.shape)
        shapey = np.hstack((shapeW[:-1], shapex[1:]))

        output_idxs = u.generate_idxs(shapey) 

        gradient = np.array([self._d_dW(idxs, shapeW) for idxs in output_idxs])
        self.gradient = gradient
        return self.gradient
    
    def _d_dW(self, out_idx, shapeW):
        W_idxs = u.generate_idxs(shapeW)
        d_dW = np.zeros(shape=shapeW)
        n = len(W_idxs[0])
        for idx in u.filter_idx(W_idxs, out_idx, n):
            x_idx = [idx[-1]]
            for out in out_idx[n-1:]:
                x_idx.append(out)
            x_idx = tuple(x_idx)            
            d_dW[idx] = self.inputs[1].value[x_idx]
        return d_dW
    


class ELU(BinaryOperation):
    def __init__(self, inputs) -> None:
        super().__init__(inputs=inputs)
        self.symbol = "ELU"
        self.value = None
        self.gradient = [None, None]

    def compute(self):
        self.value = u.elu(self.inputs[0].value, self.inputs[1].value) 
        return self.value 

    def _gradient(self):       
        self.gradient = np.array([[u.d_elu(self.inputs[0], self.inputs[1])]])
        return self.gradient
    

class LeakyReLU(BinaryOperation):
    def __init__(self, inputs) -> None:
        super().__init__(inputs=inputs)
        self.symbol = "Leaky-ReLU"
        self.value = None
        self.gradient = [None, None]

    def compute(self):
        self.value = u.leaky_relu(self.inputs[0].value, self.inputs[1].value) 
        return self.value

    def _gradient(self):        
        self.gradient = u.d_leaky_relu(self.inputs[0].value, self.inputs[1].value)
        return self.gradient




class MSE(BinaryOperation):
    pass



class CrossEntropy(BinaryOperation):
    def __init__(self, inputs) -> None:
        super().__init__(inputs=inputs)
        self.symbol = "CE"
        self.value = None
        self.gradient = [None,None]

    def compute(self):
        input_min = np.min(self.inputs[0].value)
        self.value = -np.log(self.inputs[0].value[self.inputs[1].value == 1] - input_min + 0.01)
        return self.value

    def _gradient(self):
        gradient = np.zeros(self.inputs[0].value.shape)
        mask = (self.inputs[1].value==1)
        gradient[mask==True] = -1/self.value
        self.gradient[1] = gradient.T
        return self.gradient 











def backprop_tmp(d_dj,dj_dw):
    actual = d_dj.reshape(d_dj.shape[0],-1)
    return np.tensordot(actual, dj_dw, axis=1)