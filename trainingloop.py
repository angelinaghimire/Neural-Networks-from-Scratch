import numpy as np
import config
from nn.utils import load_data, split_data
from nn.train import gradient_descent, get_predictions, get_accuracy

def main():
    print(f"Loading data from {config.DATA_PATH}...")
    data = load_data(config.DATA_PATH)
    
    print(f"Splitting data (validation size = {config.VAL_SIZE})...")
    X_train, Y_train, X_val, Y_val = split_data(data, val_size=config.VAL_SIZE)
    
    print(f"Training on {X_train.shape[1]} examples for {config.ITERATIONS} iterations with alpha={config.ALPHA}...")
    # By default, gradient_descent uses init_params_he
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, config.ITERATIONS, config.ALPHA)
    
    print("\nTraining complete.")
    
    # Evaluate on validation set
    from nn.forward import forward_prop
    _, _, _, A2_val = forward_prop(W1, b1, W2, b2, X_val)
    val_preds = get_predictions(A2_val)
    val_acc = get_accuracy(val_preds, Y_val)
    
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    print(f"Saving model to {config.MODEL_PATH}...")
    np.savez(config.MODEL_PATH, W1=W1, b1=b1, W2=W2, b2=b2)
    print("Done!")

if __name__ == "__main__":
    main()
