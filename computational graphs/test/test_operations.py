import sys
sys.path.append(sys.path[0]+'/../')

import numpy as np

import utils as u
import cgraph as cg
import operations as op

def print_gradient(operation):
    print(f"gradient of {operation.symbol} with shape {operation.gradient.shape}:", operation.gradient)

def print_compute(operation):
    inputs = [input.value for input in operation.inputs]
    print(f"compute value of {operation.symbol} with inputs {inputs} = {operation.compute()}")

###########################################################
#                       Unary                            #
###########################################################

def test_unary_Tanh_operation():
    x = cg.Input(np.random.random(size=(2,2)))
    operation = op.Tanh([x])
    print_compute(operation)
    print_gradient(operation)

def test_unary_Sigmoid_operation():
    x = cg.Input(np.random.random(size=(2,2)))
    operation = op.Sigmoid([x])
    print_compute(operation)
    print_gradient(operation)


def test_unary_L1_operation():
    x = cg.Input(np.random.random(size=(2,2)))
    operation = op.L1([x])
    print_compute(operation)
    print_gradient(operation)


def test_unary_L2_operation():
    x = cg.Input(np.random.random(size=(2,2)))
    operation = op.L2([x])
    print_compute(operation)
    print_gradient(operation)


def test_unary_ReLU_operation():
    x = cg.Input(np.random.random(size=(2,2)))
    operation = op.ReLU([x])
    print_compute(operation)
    print_gradient(operation)


###########################################################
#                       Binary                            #
###########################################################

def test_Sum_operation():
    pass


def test_TensorDot_operation():
    x=np.random.random(size=(2,1))
    #W = np.random.random(size=(1,2))
    W = np.array([[1,2]])
    dot = op.TensorDot([W,x])

    print("W\n",W)
    print("x\n",x)

    print("the result of compute: \n",dot.compute())
    print("the shape of compute result: ",dot.compute().shape)
    print("the famous gradient\n", dot.gradient, dot.gradient.shape)    
    print("reshaped gradient\n\n", dot.gradient.reshape(dot.gradient.shape[0],-1))


def test_ELU_operation():
    pass


def test_LeakyReLu_operation():
    pass


