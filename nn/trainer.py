import numpy as np
from nn.forward import forward_prop
from nn.backward import backward_prop
from nn.initialization import init_params
from nn.optimizers import get_optimizer
from nn.losses import get_loss, get_loss_grad
from nn.utils import one_hot


def get_predictions(A2):
    """Return the index (class) with the highest probability for each example."""
    return np.argmax(A2, axis=0)


def get_accuracy(predictions, Y):
    """Fraction of predictions that match the true labels."""
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, iterations, alpha,
                     input_size=784, hidden_size=10, output_size=10,
                     activation='relu', initializer='he',
                     loss='cross_entropy', optimizer='adam',
                     log_every=10, verbose=True):
    """
    Train the 2-layer network.

    Parameters
    ----------
    X           : (input_size, m) input matrix
    Y           : (m,) integer true labels
    iterations  : int   — number of gradient-descent steps
    alpha       : float — learning rate
    input_size  : int   — number of input features
    hidden_size : int   — hidden layer width
    output_size : int   — number of output classes
    activation  : str   — 'relu', 'leaky_relu', 'tanh', 'sigmoid'
    initializer : str   — 'naive', 'he', 'xavier'
    loss        : str   — 'cross_entropy', 'mse'
    optimizer   : str   — 'sgd', 'momentum', 'rmsprop', 'adam'
    log_every   : int   — print progress every N iterations (0 = silent)
    verbose     : bool  — if False, suppress all printing

    Returns
    -------
    W1, b1, W2, b2 : trained parameters
    """
    W1, b1, W2, b2 = init_params(input_size, hidden_size, output_size, initializer)
    opt = get_optimizer(optimizer, alpha)

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X, activation=activation)
        dW1, db1, dW2, db2 = backward_prop(
            Z1, A1, Z2, A2, W1, W2, X, Y,
            activation=activation, loss=loss
        )
        W1, b1, W2, b2 = opt.step(W1, b1, W2, b2, dW1, db1, dW2, db2)

        if verbose and log_every > 0 and i % log_every == 0:
            preds      = get_predictions(A2)
            acc        = get_accuracy(preds, Y)
            loss_val   = get_loss(loss, A2, one_hot(Y))
            print(f"Iter {i:4d} | Loss: {loss_val:.4f} | Acc: {acc:.4f}")

    return W1, b1, W2, b2
