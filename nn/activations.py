import numpy as np

# Apply the ReLU (Rectified Linear Unit) activation function.
def ReLU(Z):
    """
    Apply the Rectified Linear Unit (ReLU) activation function.

    ReLU introduces non-linearity into the neural network by replacing
    all negative values with 0 while leaving positive values unchanged.
    """
    return np.maximum(0,Z)

def ReLU_deriv(Z):
    """
    Compute the derivative of the ReLU activation function.

    The derivative is used during backpropagation to determine how
    gradients flow through the ReLU activation.

    Derivative:
        0 if x <= 0
        1 if x > 0
    """
    return Z > 0 

# Apply the Softmax activation function to the output layer.
def softmax(Z):
    """
    Apply the Softmax activation function.

    Softmax converts a vector of raw scores (logits) into a probability
    distribution. Each output value lies between 0 and 1, and the
    probabilities for each example sum to 1.
    """
    exp = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp / np.sum(exp, axis=0, keepdims=True)