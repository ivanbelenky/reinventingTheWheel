import operations as op
import cgraph as cg

class Layer(object):
    def __init__(self, input):
        self.input = input


class InputLayer(Layer):
    def __init__(self, input):
        super().__init__(input)
        self.out = self.input
        self.nodes = [self.out]


class WeightLayer(Layer):
    def __init__(self, input, out_shape, in_shape) -> None:
        super().__init__(input)
        self.weights = cg.Variable([out_shape, in_shape])
        self.out = op.Wx([self.weights, self.input])
        
        self.nodes = [self.weights, self.out]



class ActivationLayer(Layer):
    ACTIVATIONS = {
                        "sgm":op.Sigmoid,
                        "tanh":op.Tanh,
                        "relu":op.ReLU
                        }

    def __init__(self, input, activation):
        super().__init__(input)
        self.out = ActivationLayer.ACTIVATIONS[activation]([self.input])
        
        self.nodes = [self.out]



class ProbabilityLayer(Layer):
    PROBABILITIES = {
                        "Softmax":op.Softmax,
                        "sgm":op.Sigmoid
                    }

    def __init__(self, input, out_shape, input_shape, prob = "Softmax"):
        super().__init__(input)
        self.weights = cg.Variable([out_shape, input_shape])
        self.wx = op.Wx([self.weights, self.input])
        self.out = ProbabilityLayer.PROBABILITIES[prob]([self.wx])

        self.nodes = [self.weights, self.wx, self.out]



class LossLayer(Layer):
    LOSSES = {
                "CrossEntropy":op.CrossEntropy,
            }

    def __init__(self, input, out_shape, true_prob, loss = "CrossEntropy"):
        super().__init__(input)
        self.out = LossLayer.LOSSES[loss]([input, true_prob])
        self.nodes = [self.out]




        





