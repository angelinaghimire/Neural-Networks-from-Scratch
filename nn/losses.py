import numpy as np

def cross_entropy(A2, Y_one_hot):
    """
    Compute the Cross-Entropy loss.
    
    Parameters
    ----------
    A2 : numpy array
        Output activations (predicted class probabilities) of shape (n_classes, m).
    Y_one_hot : numpy array
        One-hot encoded true labels of shape (n_classes, m).
        
    Returns
    -------
    loss : float
    """
    m = Y_one_hot.shape[1]
    eps = 1e-15
    A2_clipped = np.clip(A2, eps, 1.0 - eps)
    return -1 / m * np.sum(Y_one_hot * np.log(A2_clipped))

def cross_entropy_grad(A2, Y_one_hot):
    """
    Derivative of Cross-Entropy loss w.r.t pre-activation Z2 (assuming Softmax output).
    dZ2 = A2 - Y
    """
    return A2 - Y_one_hot

def mse(A2, Y_one_hot):
    """
    Compute the Mean Squared Error (MSE) loss.
    
    Parameters
    ----------
    A2 : numpy array
        Output activations of shape (n_classes, m).
    Y_one_hot : numpy array
        One-hot encoded true labels of shape (n_classes, m).
        
    Returns
    -------
    loss : float
    """
    m = Y_one_hot.shape[1]
    return 1 / m * np.sum((A2 - Y_one_hot) ** 2)

def mse_grad(A2, Y_one_hot):
    """
    Derivative of MSE loss w.r.t pre-activation Z2 (assuming Softmax output).
    dZ2 = 2 * (A2 - Y) * A2 * (1 - A2)
    """
    return 2 * (A2 - Y_one_hot) * A2 * (1 - A2)

def get_loss_grad(loss_name, A2, Y_one_hot):
    """
    Retrieve pre-activation gradient for the specified loss.
    """
    if loss_name == 'mse':
        return mse_grad(A2, Y_one_hot)
    else:
        return cross_entropy_grad(A2, Y_one_hot)


def get_loss(loss_name, A2, Y_one_hot):
    """
    Compute the scalar loss value for the specified loss function.
    """
    if loss_name == 'mse':
        return mse(A2, Y_one_hot)
    else:
        return cross_entropy(A2, Y_one_hot)
