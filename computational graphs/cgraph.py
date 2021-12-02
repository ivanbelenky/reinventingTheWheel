import numpy as np
from numpy.lib.arraysetops import isin
import utils as u
import copy 



class Variable(object):
    """Variables are nodes in the computational graph that at least for now are stated as
    tensors representing the weights for all the connections related to one specific node.
    This are usually going to get connected with input or operation outputs for internal 
    layers. They have an initial value."""

    def __init__(self, shape, inputs = None, init = "default", init_max = 10E-3, range = (-1,1)):
        self.shape = shape
        self.init = init
        self.init_max = init_max
        self.range = range

        self.initial_value = self.initialize()
        self.value = self.initial_value 
        self.consumers = []

        for input_node in inputs:
            input_node.consumers.append(self)
        

    def initialize(self):
        if self.init == 'default' or self.init == 'uniform':
            return np.random.uniform(self.range[0], self.range[0], size=self.shape)/self.range
        elif self.init == 'normal':
            return np.random.normal(size=self.shape)/self.range



class Input(object):
    """Also regarded as placeholders. Here is where the data is going to be implemented as a Node"""
    
    def __init__(self, value):        
        """
        Parameters
        ----------
        value: numpy.ndarray
        
        """
        self.consumers = []
    
        # this should be smarted in order to avoid multiple init calls, maybe a proxy that inits the val
        # Node father class?

        self.value = value
        self.shape = value.shape    
        
    

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


class BinaryOperation(Operation):
    def __init__(self, inputs) -> None:
        if len(inputs) != 2:
            raise Exception(f"Invalid number of inputs for Binary Operation, given {len(inputs)}, expected 2")
        super().__init__(inputs=inputs)


class UnaryOperation(Operation):
    """Generalized Unary Operation. Before applying it sums over all elements
    of the input given, i.e. input.shape can be whatever you want"""
    
    def __init__(self, inputs) -> None:
        if len(inputs) != 1:
            raise Exception(f"Invalid number of inputs for Binary Operation, given {len(inputs)}, expected 1")
        super().__init__(inputs=inputs)
        self.input = inputs[0].value


class cGraph(object):
    """Just feedforward computational graphs class. Multiple endpoints and startpoints. 
    No way at least for now on performing gradient between two arbitrary nodes"""

    def __init__(self, nodes):
        self.nodes = nodes
        self.variables = [node for node in self.nodes if isinstance(node, Variable) ]
        self.operations = [node for node in self.nodes if isinstance(node, Operation)]
        self.input = [node for node in self.nodes if isinstance(node, Input)]

        self.start_points = self._get_start_points()
        self.end_points = self._get_end_points()


    def _get_start_points(self):
        """I did not check into the bib if this is the name they give to what I am going to define.
        Startpoints are those nodes that do not consume other nodes.

        At least for now this is redundant since variables and inputs are going to be start points."""

        return [node for node in self.nodes if isinstance(node, [Input, Variable])]

        
    def _get_end_points(self):
        """Nodes of the graph that are not getting consumed by no body."""
        return [node for node in self.nodes if not node.consumers]

    def get_Output_Variable_routes(self):
        """At least for now the gradients are always and only calculated regarding 
        endpoints and variables. """
        
        routes_list = []
        for endpoint in self.end_points:
            for startpoint in self.start_points:
                if isinstance(startpoint, Variable): #just added this line since maybe someday well be old, being realistic, maybe I get some generalized non directed route
                    routes = []
                    actual_route = []
                    self._get_routes(endpoint, startpoint, routes, actual_route)
                    if routes:
                        routes_list.append(copy.deepcopy(routes))

        return routes 

    def get_routes(self, start, end, routes, actual_route):
        actual_route.append(start)

        if start == end:
            routes.append(copy.deepcopy(actual_route))
            actual_route.remove(start)
            return 

        elif not start.consumers:
            actual_route.remove(start)
            return 

        else:
            for consumer in start.consumers:
                self.get_routes(consumer, end, routes, actual_route)

        actual_route.remove(start)

        return 












