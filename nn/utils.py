import numpy as np
import pandas as pd

def load_data(path):
    """
    Load the MNIST CSV dataset.

    Returns
    -------
    data : numpy array of shape (m, 785)
           Column 0 is the label; columns 1-784 are pixel values.
    """
    df = pd.read_csv(path)
    return np.array(df)


def split_data(data, val_size=1000):
    """
    Shuffle and split raw data into train/validation sets.

    Parameters
    ----------
    data     : numpy array (m, 785) — label in column 0, pixels in columns 1-784
    val_size : number of examples to reserve for validation

    Returns
    -------
    X_train : (784, m_train)  — pixel values, normalised to [0, 1]
    Y_train : (m_train,)      — integer labels
    X_val   : (784, val_size) — pixel values, normalised to [0, 1]
    Y_val   : (val_size,)     — integer labels
    """
    np.random.shuffle(data)

    m, n = data.shape  # n = 785

    # --- validation split ---
    val      = data[:val_size].T
    Y_val    = val[0].astype(int)
    X_val    = val[1:n] / 255.0

    # --- training split ---
    train    = data[val_size:].T
    Y_train  = train[0].astype(int)
    X_train  = train[1:n] / 255.0

    return X_train, Y_train, X_val, Y_val

def one_hot(Y):
    """Convert an integer label vector into a one-hot matrix (n_classes x m)."""
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T
