import numpy as np
from numpy.lib.arraysetops import isin
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


class BinaryOperation(Operation):
    def __init__(self, inputs) -> None:
        if len(inputs) != 1:
            raise Exception(f"Invalid number of inputs for Binary Operation, given {len(inputs)}, expected 2")
        super().__init__(inputs=inputs)


class UnaryOperation(Operation):
    """Generalized Unary Operation. Before applying it sums over all elements
    of the input given, i.e. input.shape can be whatever you want"""
    
    def __init__(self, inputs) -> None:
        if len(inputs) != 1:
            raise Exception(f"Invalid number of inputs for Binary Operation, given {len(inputs)}, expected 1")
        super().__init__(inputs=inputs)
    


    





