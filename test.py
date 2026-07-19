import numpy as np
import config
from nn.utils import load_data, split_data
from nn.predict import test_prediction, make_predictions
from nn.trainer import get_accuracy

def main():
    print(f"Loading data from {config.DATA_PATH}...")
    try:
        data = load_data(config.DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {config.DATA_PATH}")
        return

    _, _, X_val, Y_val = split_data(data, val_size=config.VAL_SIZE)
    
    print(f"Loading model from {config.MODEL_PATH}...")
    try:
        model = np.load(config.MODEL_PATH)
        W1 = model['W1']
        b1 = model['b1']
        W2 = model['W2']
        b2 = model['b2']
    except FileNotFoundError:
        print(f"Error: Model file {config.MODEL_PATH} not found. Run train.py first.")
        return

    # Evaluate on validation set
    print("Evaluating validation accuracy...")
    val_preds = make_predictions(X_val, W1, b1, W2, b2, activation=config.ACTIVATION)
    val_acc = get_accuracy(val_preds, Y_val)
    print(f"Validation Accuracy: {val_acc:.4f}\n")

    # Show a few interactive predictions
    print("Showing 3 random validation predictions:")
    for _ in range(3):
        idx = np.random.randint(0, X_val.shape[1])
        test_prediction(idx, X_val, Y_val, W1, b1, W2, b2, activation=config.ACTIVATION)

if __name__ == "__main__":
    main()
