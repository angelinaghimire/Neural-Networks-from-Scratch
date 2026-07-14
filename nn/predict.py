import numpy as np

from nn.forward import forward_prop
from nn.train import get_predictions


def make_predictions(X, W1, b1, W2, b2):
    """Run forward pass and return predicted class for each example in X."""
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    return get_predictions(A2)


def test_prediction(index, X_train, Y_train, W1, b1, W2, b2):
    """
    Print the image at `index` to the terminal using ASCII art, 
    and show its true label and the model's prediction.

    Parameters
    ----------
    index    : column index into X_train
    X_train  : (784, m) training data
    Y_train  : (m,)    true labels
    W1,b1,W2,b2 : trained parameters
    """
    current_image = X_train[:, index, None]
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = Y_train[index]

    print(f"\nModel Prediction: {prediction[0]}")
    print(f"True Label:       {label}")

    img = current_image.reshape((28, 28))
    
    # ASCII characters from light to dark
    chars = " .:-=+*#%@"
    
    print("-" * 30)
    for row in img:
        row_str = "|"
        for px in row:
            # px is between 0 and 1, map it to an index 0-9
            idx = int(px * 9.99)
            row_str += chars[idx]
        row_str += "|"
        print(row_str)
    print("-" * 30)
