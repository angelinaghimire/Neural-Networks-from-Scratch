def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    """
    Update the network parameters using gradient descent.
    
    Parameters
    ----------
    W1 : numpy array
        Weights of the hidden layer.
    b1 : numpy array
        Biases of the hidden layer.
    W2 : numpy array
        Weights of the output layer.
    b2 : numpy array
        Biases of the output layer.
    dW1, db1, dW2, db2 : numpy arrays
        Gradients for the corresponding parameters.
    alpha : float
        The learning rate.
        
    Returns
    -------
    W1, b1, W2, b2 : numpy arrays
        The updated parameters.
    """
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2