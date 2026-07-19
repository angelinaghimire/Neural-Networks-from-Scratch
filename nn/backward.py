import numpy as np
from nn.utils import one_hot
from nn.activations import get_activation_deriv
from nn.losses import get_loss_grad


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y,
                  activation='relu', loss='cross_entropy'):
    """
    Backward propagation — compute gradients for all parameters.

    Parameters
    ----------
    Z1, A1     : Pre-activations and activations of the hidden layer.
    Z2, A2     : Pre-activations and activations of the output layer.
    W1, W2     : Weight matrices.
    X          : Input data of shape (input_size, m).
    Y          : True integer labels of shape (m,).
    activation : Hidden layer activation name ('relu', 'leaky_relu', 'tanh', 'sigmoid').
    loss       : Loss function name ('cross_entropy', 'mse').

    Returns
    -------
    dW1, db1, dW2, db2 : Gradients w.r.t. all parameters.
    """
    m = X.shape[1]
    one_hot_Y = one_hot(Y)

    # Output layer gradient (depends on chosen loss)
    dZ2 = get_loss_grad(loss, A2, one_hot_Y)     # (output, m)

    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    # Hidden layer gradient (depends on chosen activation)
    act_deriv = get_activation_deriv(activation)
    dZ1 = W2.T.dot(dZ2) * act_deriv(Z1)         # (hidden, m)

    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2