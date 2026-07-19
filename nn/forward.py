from nn.activations import get_activation, softmax


def forward_prop(W1, b1, W2, b2, X, activation='relu'):
    """
    Forward propagation through the 2-layer network.

    Parameters
    ----------
    W1, b1     : Hidden layer weights and biases.
    W2, b2     : Output layer weights and biases.
    X          : Input data of shape (input_size, m).
    activation : Hidden layer activation name ('relu', 'leaky_relu', 'tanh', 'sigmoid').

    Returns
    -------
    Z1, A1, Z2, A2 : Pre-activations and activations for both layers.
    """
    act_fn = get_activation(activation)

    Z1 = W1.dot(X) + b1          # (hidden, m)
    A1 = act_fn(Z1)               # apply chosen activation
    Z2 = W2.dot(A1) + b2          # (output, m)
    A2 = softmax(Z2)              # probability distribution over classes
    return Z1, A1, Z2, A2