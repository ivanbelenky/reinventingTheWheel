import sys
sys.path.append(sys.path[0]+'/../')

import numpy as np

import utils as u
import cgraph as cg


def test_Dot_operation():
    x=np.random.random(size=(2,2))
    #W = np.random.random(size=(1,2))
    W = np.array([[1,2],[3,1]])
    dot = cg.TensorDot([W,x])

    print("W\n",W)
    print("x\n",x)

    print("the result of compute: \n",dot.compute())
    print("the shape of compute result: ",dot.compute().shape)
    print("the famous gradient\n", dot.gradient, dot.gradient.shape)    
    print("reshaped gradient\n\n", dot.gradient.reshape(dot.gradient.shape[0],-1))

if __name__ == "__main__":
    test_Dot_operation()
