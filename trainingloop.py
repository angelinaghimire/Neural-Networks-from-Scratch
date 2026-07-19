import numpy as np
import config
from nn.utils import load_data, split_data
from nn.forward import forward_prop
from nn.trainer import gradient_descent, get_predictions, get_accuracy


def main():
    print(f"Loading data from  : {config.DATA_PATH}")
    data = load_data(config.DATA_PATH)

    print(f"Validation size    : {config.VAL_SIZE}")
    X_train, Y_train, X_val, Y_val = split_data(data, val_size=config.VAL_SIZE)

    print(
        f"\nConfig:"
        f"\n  input={config.INPUT_SIZE}  hidden={config.HIDDEN_SIZE}  output={config.OUTPUT_SIZE}"
        f"\n  activation={config.ACTIVATION}  initializer={config.INITIALIZER}"
        f"\n  loss={config.LOSS}  optimizer={config.OPTIMIZER}"
        f"\n  lr={config.ALPHA}  iterations={config.ITERATIONS}"
        f"\n  train samples={X_train.shape[1]}\n"
    )

    W1, b1, W2, b2 = gradient_descent(
        X_train, Y_train,
        iterations  = config.ITERATIONS,
        alpha       = config.ALPHA,
        input_size  = config.INPUT_SIZE,
        hidden_size = config.HIDDEN_SIZE,
        output_size = config.OUTPUT_SIZE,
        activation  = config.ACTIVATION,
        initializer = config.INITIALIZER,
        loss        = config.LOSS,
        optimizer   = config.OPTIMIZER,
    )

    print("\n── Validation ──────────────────────────────")
    _, _, _, A2_val = forward_prop(W1, b1, W2, b2, X_val, activation=config.ACTIVATION)
    val_preds = get_predictions(A2_val)
    val_acc   = get_accuracy(val_preds, Y_val)
    print(f"Validation Accuracy: {val_acc:.4f}")

    print(f"\nSaving model to {config.MODEL_PATH} ...")
    np.savez(config.MODEL_PATH, W1=W1, b1=b1, W2=W2, b2=b2)
    print("Done!")


if __name__ == "__main__":
    main()
