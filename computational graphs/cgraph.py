import numpy as np
import utils as u

class Operation(object):
    """"""

    def __init__(self, inputs = None) -> None:
        self.inputs = inputs if inputs else []
        self.consumers = []

        for input_node in inputs:
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



class Dot(Operation):
    def __init__(self, inputs) -> None:
        super().__init__(inputs=inputs)
        self.symbol = "*"
        self._gradient()

    def compute(self):
        return self.input[0].dot(self.inputs[1])

    @property
    def gradient(self):
        return self.__gradient

    def _gradient(self):
        shapeW = np.array(self.inputs[0].shape)[:-1]
        shapex = np.array(self.inputs[1].shape)[1:]
        shapey = np.hstack((shapeW, shapex))

        output_idxs = u.generate_idxs(shapey) 
        gradient = [self._d_dW(idxs, shapeW) for idxs in output_idxs]

        self.__gradient = gradient
    
    def _d_dW(self, out_idxs, shapeW):
        W_idxs = u.generate_idxs(shapeW)
        d_dW = np.zeros(size=shapeW)
        for idx in W_idxs:
            n = W_idxs.shape[0]
            x_idx = tuple([idx[-1]].append(out_idxs[n:]))
            d_dW[idx] = self.inputs[0][x_idx]
        return d_dW


    def gradient(self, idx):
        return self.gradient[idx]
        

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

    





