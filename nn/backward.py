import numpy as np
from nn.utils import one_hot
from nn.activations import ReLU_deriv

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    """
    Perform the backward propagation step to compute gradients.
    
    Parameters
    ----------
    Z1 : Pre-activations of the hidden layer.
    A1 : Activations of the hidden layer.
    Z2 : Pre-activations of the output layer.
    A2 : Activations (predictions) of the output layer.
    W1 : Weights of the hidden layer.
    W2 : Weights of the output layer.
    X  : Input data.
    Y  : True labels.
        
    Returns
    -------
    dW1 : Gradient of the loss with respect to W1.
    db1 : Gradient of the loss with respect to b1.
    dW2 : Gradient of the loss with respect to W2.
    db2 : Gradient of the loss with respect to b2.
    """
    m = X.shape[1]
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2