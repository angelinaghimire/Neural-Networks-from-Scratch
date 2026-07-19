import numpy as np

def naive_init(input_size, hidden_size, output_size):
    """
    Naive random initialization.
    Draws weights from N(0,1) without scaling. Only suitable for quick experiments.
    """
    W1 = np.random.randn(hidden_size, input_size)
    b1 = np.random.randn(hidden_size, 1)
    W2 = np.random.randn(output_size, hidden_size)
    b2 = np.random.randn(output_size, 1)
    return W1, b1, W2, b2


def he_init(input_size, hidden_size, output_size):
    """
    He (Kaiming) initialization.
    Scales weights by sqrt(2 / fan_in). Recommended for ReLU / Leaky ReLU networks.
    Keeps pre-activation variance ≈ 1 across layers.
    """
    W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2


def xavier_init(input_size, hidden_size, output_size):
    """
    Xavier (Glorot) initialization.
    Scales weights by sqrt(1 / fan_in). Recommended for Sigmoid / Tanh networks.
    Balances variance in forward and backward passes.
    """
    W1 = np.random.randn(hidden_size, input_size) * np.sqrt(1.0 / input_size)
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * np.sqrt(1.0 / hidden_size)
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2


_INITIALIZERS = {
    'naive':   naive_init,
    'he':      he_init,
    'xavier':  xavier_init,
}


def init_params(input_size, hidden_size, output_size, initializer='he'):
    """
    Dispatcher: initialise network weights by name.

    Parameters
    ----------
    input_size  : int
    hidden_size : int
    output_size : int
    initializer : str — 'naive', 'he', or 'xavier'

    Returns
    -------
    W1, b1, W2, b2 : numpy arrays
    """
    if initializer not in _INITIALIZERS:
        raise ValueError(f"Unknown initializer '{initializer}'. Choose from: {list(_INITIALIZERS)}")
    return _INITIALIZERS[initializer](input_size, hidden_size, output_size)