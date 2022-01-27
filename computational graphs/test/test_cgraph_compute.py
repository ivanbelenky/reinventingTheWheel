import sys
sys.path.append(sys.path[0]+'/../')

import numpy as np

import utils as u
import cgraph as cg
import operations as op


W1 = cg.Variable([100,3072])
x = cg.Input(-0.5+np.random.random(size=(3072,1)))
W1x = op.Wx([W1,x])

b = cg.Variable((100,1), range=(0E-3,1E-3))
W1x_b = op.Sum([W1x,b])

#(W1.x+b)

#act = op.Sigmoid([W1x_b])
relu_slope = cg.Variable([1,1],range=(1E-2,1E-3))
act = op.LeakyReLU([relu_slope, W1x_b])

W2 = cg.Variable([10,100])
W2y = op.Wx([W2,act])
y_t = cg.Input(-0.5+np.random.random(size=(10,1)))
minus = op.Sum([W2y, y_t])
cost = op.L2([minus])

# L2(W2.sgm(W1.x+b)-y_t)

nodes = [W1, x, W1x, W1x_b, act, W2, W2y, y_t, minus, cost]
graph = cg.cGraph(nodes)
graph.compute()

gradients = graph.gradient(cost,relu_slope)


print('\n al \n', [g.shape for g in gradients[0]])



def propagate_gradient(gradients_list):
    total_grad = 0
    for gradients in gradients_list:
        grad_0 = gradients[0]
        for grad in gradients[1:-1]:
            grad_0 = grad_0.dot(grad)

        if gradients[-1].shape[0] != 1:
            grad_0 = grad_0.dot(gradients[-1])
            total_grad += grad_0.T

        else:
            grad_0 = grad_0.T.dot(gradients[-1])
            total_grad += grad_0

    return total_grad


grad = propagate_gradient(gradients)

