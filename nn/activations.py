import numpy as np

# Hidden-layer activations

def relu(Z):
    """ReLU: max(0, Z). Fast, avoids vanishing gradients. Default choice."""
    return np.maximum(0, Z)

def relu_deriv(Z):
    """Derivative of ReLU."""
    return Z > 0

def leaky_relu(Z, alpha=0.01):
    """Leaky ReLU: prevents dying neurons by allowing small negative slope."""
    return np.where(Z > 0, Z, alpha * Z)

def leaky_relu_deriv(Z, alpha=0.01):
    """Derivative of Leaky ReLU."""
    return np.where(Z > 0, 1.0, alpha)

def tanh(Z):
    """Tanh: output in (-1, 1). Works well with Xavier init."""
    return np.tanh(Z)

def tanh_deriv(Z):
    """Derivative of Tanh."""
    return 1 - np.tanh(Z) ** 2

def sigmoid(Z):
    """Sigmoid: output in (0, 1). Can suffer from vanishing gradients."""
    Z = np.clip(Z, -500, 500)
    return 1 / (1 + np.exp(-Z))

def sigmoid_deriv(Z):
    """Derivative of Sigmoid."""
    sig = sigmoid(Z)
    return sig * (1 - sig)

# Output-layer activation

def softmax(Z):
    """
    Softmax: converts logits to class probabilities.
    Numerically stable via max-subtraction.
    """
    exp = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp / np.sum(exp, axis=0, keepdims=True)

# Helpers — used by forward.py and backward.py

_ACTIVATIONS = {
    'relu':       (relu,       relu_deriv),
    'leaky_relu': (leaky_relu, leaky_relu_deriv),
    'tanh':       (tanh,       tanh_deriv),
    'sigmoid':    (sigmoid,    sigmoid_deriv),
}

def get_activation(name):
    """Return the activation function for the given name."""
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[name][0]

def get_activation_deriv(name):
    """Return the activation derivative for the given name."""
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[name][1]

# Legacy aliases (kept for any external code that imports them directly)
ReLU        = relu
ReLU_deriv  = relu_deriv