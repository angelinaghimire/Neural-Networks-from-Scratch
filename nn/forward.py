from nn.activations import ReLU, softmax

def forward_prop(W1, b1, W2, b2, X):
    """
    Perform the forward propagation step through the network.
    
    Parameters
    ----------
    W1 : numpy array of shape (n_hidden, n_features)
        Weights of the first (hidden) layer.
    b1 : numpy array of shape (n_hidden, 1)
        Biases of the first layer.
    W2 : numpy array of shape (n_classes, n_hidden)
        Weights of the second (output) layer.
    b2 : numpy array of shape (n_classes, 1)
        Biases of the second layer.
    X  : numpy array of shape (n_features, m)
        The input data, where each column is an example.
        
    Returns
    -------
    Z1 : numpy array
        Pre-activations for the hidden layer.
    A1 : numpy array
        Activations (ReLU) for the hidden layer.
    Z2 : numpy array
        Pre-activations for the output layer.
    A2 : numpy array
        Activations (Softmax) for the output layer.
    """
    # Compute the weighted sum for the hidden layer.
    Z1 = W1.dot(X) + b1
    # Apply the ReLU activation function.
    A1 = ReLU(Z1)
    # Compute the weighted sum for the output layer.
    Z2 = W2.dot(A1) + b2
    # Apply the Softmax activation function to convert the output scores into probabilities for each class.
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2