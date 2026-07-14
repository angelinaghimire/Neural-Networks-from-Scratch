import numpy as np
from nn.forward import forward_prop
from nn.backward import backward_prop
from nn.update import update_params
from nn.initialization import init_params


def get_predictions(A2):
    """Return the index (class) with the highest probability for each example."""
    return np.argmax(A2, axis=0)


def get_accuracy(predictions, Y):
    """Fraction of predictions that match the true labels."""
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, iterations, alpha, init_fn=init_params):
    """
    Train the network with gradient descent.

    Parameters
    ----------
    X          : (784, m) input matrix
    Y          : (m,)    true labels
    iterations : number of gradient descent steps
    alpha      : learning rate
    init_fn    : initialization function to use
    """
    W1, b1, W2, b2 = init_fn()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 10 == 0:
            preds = get_predictions(A2)
            acc = get_accuracy(preds, Y)
            print(f"Iteration {i:4d} | Accuracy: {acc:.4f}")

    return W1, b1, W2, b2
