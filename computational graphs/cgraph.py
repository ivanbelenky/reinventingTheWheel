import numpy as np
from numpy.lib.arraysetops import isin
import utils as u
import copy 
import uuid



class Variable(object):
    """Variables are nodes in the computational graph that at least for now are stated as
    tensors representing the weights for all the connections related to one specific node.
    This are usually going to get connected with input or operation outputs for internal 
    layers. They have an initial value."""

    def __init__(self, shape, init = "default", init_max = 1E-3, range = (-1,1)):
        self.id = uuid.uuid4()
        
        self.shape = shape
        self.init = init
        self.init_max = init_max
        self.range = range

        self.initial_value = self.initialize()
        self.value = self.initial_value 
        self.consumers = []
        

    def initialize(self):
        if self.init == 'default' or self.init == 'uniform':
            return np.random.uniform(low=self.range[0], high=self.range[1], size=self.shape)*self.init_max
        elif self.init == 'normal':
            return np.random.normal(size=self.shape)



class Input(object):
    """Also regarded as placeholders. Here is where the data is going to be implemented as a Node"""
    
    def __init__(self, value=None):        
        """
        Parameters
        ----------
        value: numpy.ndarray
        
        """
        self.consumers = []
        self.id = uuid.uuid4()        

        self.value = value
        
    

class Operation(object):
    """"""

    def __init__(self, inputs = None) -> None:
        self.inputs = inputs if inputs else []
        self.consumers = []
        self.id = uuid.uuid4()
        

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
        self.input = inputs[0]
        self.value = None
        self.gradient = None

class cGraph(object):
    """Just feedforward computational graphs class. Multiple endpoints and startpoints. 
    No way at least for now on performing gradient between two arbitrary nodes"""

    def __init__(self, nodes):
        self.nodes = nodes
        self.variables = [node.id for node in self.nodes if isinstance(node, Variable) ]
        self.operations_id = [node.id for node in self.nodes if isinstance(node, Operation)]
        self.input = [node.id for node in self.nodes if isinstance(node, Input)]

        self.operations = [node for node in self.nodes if isinstance(node, Operation)]
        self.start_points = self._get_start_points()
        self.end_points = self._get_end_points()


    def _get_start_points(self):
        """I did not check into the bib if this is the name they give to what I am going to define.
        Startpoints are those nodes that do not consume other nodes.

        At least for now this is redundant since variables and inputs are going to be start points."""

        return [node for node in self.nodes if isinstance(node, (Input, Variable))]

        
    def _get_end_points(self):
        """Nodes of the graph that are not getting consumed by no body."""
        return [node for node in self.nodes if not node.consumers]

    def get_Output_Variable_routes(self):
        """At least for now the gradients are always and only calculated regarding 
        endpoints and variables. """
        
        routes_list = []
        for endpoint in self.end_points:
            for startpoint in self.start_points: 
                routes = []
                actual_route = []
                self.get_routes(startpoint, endpoint, routes, actual_route)
                if routes:
                    routes_list.append(copy.deepcopy(routes))

        return routes_list


    def get_routes(self, start, end, routes, actual_route):
        actual_route.append(start.id)

        if start == end:
            routes.append(copy.deepcopy(actual_route))
            actual_route.remove(start.id)
            return 

        elif not start.consumers:
            actual_route.remove(start.id)
            return 

        else:
            for consumer in start.consumers:
                self.get_routes(consumer, end, routes, actual_route)

        actual_route.remove(start.id)

        return 


    def compute_order(self):
        self.order_table = [0 for operation in self.operations_id]
        all_routes = self.get_Output_Variable_routes()
        for paths_list in all_routes:
            for path in paths_list:
                auxpath = path[1:]
                for i,node in enumerate(auxpath):
                    idx = self.operations_id.index(node)
                    if self.order_table[idx] < i:
                        self.order_table[idx] = i
        
        return self.order_table


    def compute(self):
        orders = self.compute_order()
        depths = [] 

        for i in range(0,max(orders)+1):
            layer = []
            for operation, depth in zip(self.operations,orders):
                if depth == i:
                    layer.append(operation)
            depths.append(layer) 


        for layer in depths:
            for operation in layer:
                operation.compute()
            
            for operation in layer:
                operation._gradient()


    def get_outputs(self):
        return [end.value for end in self.end_points]



    def gradient(self,out,var):

        import operations as op

        if not isinstance(var, Variable):
            raise Exception("Not implemented general differentiation")
        if out not in self.end_points:
            raise Exception("Not implemented general differentiation")

        routes = []
        ar = []
        self.get_routes(var, out, routes, ar)

        gradient_list = []
        if routes:
            for route in routes:
                gradient = []
                for operation_id in route[::-1][:-2]:
                    operation = self.find_operation(operation_id)

                    if isinstance(operation, BinaryOperation):
                        gradient.append(operation.gradient[1])
                    else:
                        gradient.append(operation.gradient)

                operation = self.find_operation(route[1])
                if isinstance(operation, BinaryOperation):
                    gradient.append(operation.gradient[0])
                
                gradient_list.append(gradient)

        return gradient_list

    
    def propagate_gradient(self, gradients_list):
        total_grad = 0

        for gradients in gradients_list:
            if gradients:
                grad_0 = gradients[0]
                for grad in gradients[1:-1]:
                    grad_0 = grad_0.dot(grad)

                if gradients[-1].shape[0] != 1:
                    grad_0 = grad_0.dot(gradients[-1])
                    total_grad += grad_0.T

                else:
                    grad_0 = grad_0.reshape(1,-1)
                    grad_0 = grad_0.T.dot(gradients[-1])
                    total_grad += grad_0

        return total_grad


    def find_operation(self, operation_id):
        for operation in self.operations:
            if operation.id == operation_id:
                return operation


        




        
        
        











