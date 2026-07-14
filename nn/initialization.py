import numpy as np

# def init_params_he():
#     """
#     He initialization — recommended for ReLU networks.

#     Scales weights by sqrt(2 / fan_in) so pre-activations have
#     unit variance even after passing through many inputs.

#     With 784 inputs:
#         W1 std dev = sqrt(2/784) ≈ 0.050  →  Z1 std dev ≈ 1.0  ✓
#     """
#     W1 = np.random.randn(10, 784) * np.sqrt(2 / 784)
#     b1 = np.zeros((10, 1))
#     W2 = np.random.randn(10, 10) * np.sqrt(2 / 10)
#     b2 = np.zeros((10, 1))
#     return W1, b1, W2, b2


def init_params():
    """
    Naive random initialization.
    
    Initializes weights and biases for a 2-layer neural network from a 
    standard normal distribution (mean=0, std=1) without any scaling.
    
    Returns
    -------
    W1 : numpy array of shape (10, 784)
        Weights for the hidden layer.
    b1 : numpy array of shape (10, 1)
        Biases for the hidden layer.
    W2 : numpy array of shape (10, 10)
        Weights for the output layer.
    b2 : numpy array of shape (10, 1)
        Biases for the output layer.
    """
    W1 = np.random.randn(10, 784)
    b1 = np.random.randn(10, 1)
    W2 = np.random.randn(10, 10)
    b2 = np.random.randn(10, 1)
    return W1, b1, W2, b2