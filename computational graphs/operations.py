from cgraph import UnaryOperation, BinaryOperation
from variables import Variable

import utils as u
import numpy as np



###########################################################
#                       Unary                            #
###########################################################


class Tanh(UnaryOperation):
    def __init__(self, inputs) -> None:
        super().__init__(inputs=inputs)
        self.symbol = "tanh"
        self.value = self.inputs[0].value

    @property
    def gradient(self):
        self._gradient()
        return self.__gradient

    def compute(self):
        return np.tanh(self.value)

    def _gradient(self):       
        gradient = np.array([[1-(np.tanh(self.value))**2]])
        self.__gradient = gradient


class Sigmoid(UnaryOperation):
    def __init__(self, inputs) -> None:
        super().__init__(inputs=inputs)
        self.symbol = "sgm"
        self.value = self.inputs[0].value

    @property
    def gradient(self):
        self._gradient()
        return self.__gradient

    def compute(self):
        return u.sgm(self.value)

    def _gradient(self):       
        gradient = np.array([[u.d_sgm(self.value)]])
        self.__gradient = gradient


class L1(UnaryOperation):
    def __init__(self, inputs) -> None:
        super().__init__(inputs)
        self.symbol = "L1"
        self.value = self.inputs[0].value
    
    @property
    def gradient(self):
        self._gradient()
        return self.__gradient

    def compute(self):
        return np.linalg.norm(self.value.reshape(-1),1)
    
    def _gradient(self):
        gradient = np.zeros(shape=self.value.shape)
        x_idxs = u.generate_idxs(self.value.shape)
        _L2 = self.compute()
        gradient = self.value/_L2
        self.__gradient = np.array([[gradient]])

class L2(UnaryOperation):
    def __init__(self, inputs) -> None:
        super().__init__(inputs)
        self.symbol = "L2"
        self.value = self.inputs[0].value
    
    @property
    def gradient(self):
        self._gradient()
        return self.__gradient

    def compute(self):
        return np.linalg.norm(self.value)
    
    def _gradient(self):
        gradient = np.zeros(shape=self.value.shape)
        x_idxs = u.generate_idxs(self.value.shape)
        _L2 = self.compute()
        gradient = self.value/_L2
        self.__gradient = np.array([[gradient]])


class ReLU(UnaryOperation):
    def __init__(self, inputs) -> None:
        super().__init__(inputs=inputs)
        self.symbol = "ReLU"
        self.value = self.inputs[0].value

    @property
    def gradient(self):
        self._gradient()
        return self.__gradient

    def compute(self):
        return u.relu(self.value)

    def _gradient(self):       
        gradient = np.array([[u.d_relu(self.value)]])
        self.__gradient = gradient
        

###########################################################
#                       Binary                            #
###########################################################


class Sum(BinaryOperation):
    def __init__(self, inputs) -> None:
        super().__init__(inputs=inputs)
        self.symbol = "+"
        self._gradient()

    def compute(self):
        return np.sum(self.inputs)        

    @property
    def gradient(self):
        return self._gradient

    def _gradient(self):
        shapes = np.array([input.shape for input in self.shape])
        if np.unique(shapes).shape[0] != 1:
            raise Exception("Sum with inputs of different shapes")        

        shape = self.inputs[0].value.shape
        self._gradient = np.ones(shape)

    def gradient(self, idx):
        # if idx < 0 or idx > len(self.inputs)-1:
        #     raise Exception("Invalid idx")
        return self.gradient


class TensorDot(BinaryOperation):
    def __init__(self, inputs) -> None:
        super().__init__(inputs=inputs)
        self.symbol = "*"
        self._gradient()


    def compute(self):
        W = self.inputs[0].value
        x = self.inputs[1].value
        return np.tensordot(W,x,axes=1) 

    @property
    def gradient(self):
        return self.__gradient

    def _gradient(self):
        shapeW = np.array(self.inputs[0].value.shape)
        shapex = np.array(self.inputs[1].value.shape)
        shapey = np.hstack((shapeW[:-1], shapex[1:]))

        output_idxs = u.generate_idxs(shapey) 

        gradient = np.array([self._d_dW(idxs, shapeW) for idxs in output_idxs])
        self.__gradient = gradient
    
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
        self.x = np.sum(self.inputs[0].value)
        self.a = self.inputs[1].value
        if not isinstance(self.a, Variable):
            raise Exception("a constant of ELU must be a parameter")

    @property
    def gradient(self):
        return self.__gradient

    def compute(self):
        return u.elu(self.x, self.a)

    def _gradient(self):       
        gradient = np.array([[u.d_elu(self.x, self.a)]])
        return gradient
    

class LeakyReLU(BinaryOperation):
    def __init__(self, inputs) -> None:
        super().__init__(inputs=inputs)
        self.symbol = "Leaky-ReLU"
        self.x = np.sum(self.inputs[0].value)
        self.a = self.inputs[1].value
        if not isinstance(self.a, Variable):
            raise Exception("a constant of ELU must be a parameter")

    @property
    def gradient(self):
        return self.__gradient

    def compute(self):
        return u.leaky_relu(self.x, self.a)

    def _gradient(self):       
        gradient = np.array([[u.d_leaky_relu(self.x, self.a)]])
        return gradient




class MSE(BinaryOperation):
    pass



class CrossEntropy(BinaryOperation):
    pass



class Softmax(BinaryOperation):
    pass








def backprop_tmp(d_dj,dj_dw):
    actual = d_dj.reshape(d_dj.shape[0],-1)
    return np.tensordot(actual, dj_dw, axis=1)