import numpy as np
import utils as u

class Operation(object):
    """"""

    def __init__(self, inputs = None) -> None:
        self.inputs = inputs if inputs else []
        self.consumers = []

        for input_node in inputs:
            if isinstance(input_node, Operation):
                input_node.consumers.append(self)
        
    def compute(self):
        pass

    def gradient(self, idx):
        pass
    

class Sum(Operation):
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

        shape = self.inputs[0].shape
        self._gradient = np.ones(shape)

    def gradient(self, idx):
        # if idx < 0 or idx > len(self.inputs)-1:
        #     raise Exception("Invalid idx")
        return self.gradient



class TensorDot(Operation):
    def __init__(self, inputs) -> None:
        super().__init__(inputs=inputs)
        self.symbol = "*"
        self._gradient()

    def compute(self):
        W = self.inputs[0]
        x = self.inputs[1]
        return np.tensordot(W,x,axes=1) 

    @property
    def gradient(self):
        return self.__gradient

    def _gradient(self):
        shapeW = np.array(self.inputs[0].shape)
        shapex = np.array(self.inputs[1].shape)
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
            d_dW[idx] = self.inputs[1][x_idx]
        return d_dW
        



class Tanh(Operation):
    pass

class Sigmoid(Operation):
    pass

class RELU(Operation):
    pass

class ELU(Operation):
    pass
    
class LeakyRELU(Operation):
    pass

class L2(Operation):
    pass

class L1(Operation):
    pass

class MSE(Operation):
    pass

class CrossEntropy(Operation):
    pass

class Softmax(Operation):
    pass

    





