"""
experiment.py — Grid search over activations, initializers, losses, and optimizers.

Trains each combination for a fixed number of iterations, then prints and saves
a results table sorted by validation accuracy.

Usage:
    python3 experiment.py
"""

import numpy as np
import config
from nn.utils import load_data, split_data
from nn.forward import forward_prop
from nn.trainer import gradient_descent, get_predictions, get_accuracy

# ── Grid definition ──────────────────────────────────────────────────────────
GRID = {
    "activation":  ["relu", "leaky_relu", "tanh", "sigmoid"],
    "initializer": ["he",   "xavier",     "naive"],
    "loss":        ["cross_entropy", "mse"],
    "optimizer":   ["sgd",  "momentum",  "rmsprop", "adam"],
}

# Recommended pairings (activation → sensible initializer default).
# Experiments will only run (activation, initializer) pairs on this list
# so we avoid known poor combos (e.g. relu + xavier) and reduce runtime.
SENSIBLE_PAIRS = {
    "relu":       ["he",    "naive"],
    "leaky_relu": ["he",    "naive"],
    "tanh":       ["xavier","naive"],
    "sigmoid":    ["xavier","naive"],
}

EXPERIMENT_ITERATIONS = 200
ALPHA                 = 0.001   # works well with Adam/RMSProp; SGD may need more
HIDDEN_SIZE           = 64
LOG_EVERY             = 0       # suppress per-iteration logging during grid search


def run_experiments(X_train, Y_train, X_val, Y_val):
    results = []

    total = sum(
        len(SENSIBLE_PAIRS[act]) * len(GRID["loss"]) * len(GRID["optimizer"])
        for act in GRID["activation"]
    )
    done = 0

    for activation in GRID["activation"]:
        for initializer in SENSIBLE_PAIRS[activation]:
            for loss in GRID["loss"]:
                for optimizer in GRID["optimizer"]:
                    done += 1
                    tag = f"{activation}/{initializer}/{loss}/{optimizer}"
                    print(f"[{done:3d}/{total}] {tag} ...", end="", flush=True)

                    try:
                        W1, b1, W2, b2 = gradient_descent(
                            X_train, Y_train,
                            iterations  = EXPERIMENT_ITERATIONS,
                            alpha       = ALPHA,
                            input_size  = config.INPUT_SIZE,
                            hidden_size = HIDDEN_SIZE,
                            output_size = config.OUTPUT_SIZE,
                            activation  = activation,
                            initializer = initializer,
                            loss        = loss,
                            optimizer   = optimizer,
                            log_every   = LOG_EVERY,
                            verbose     = False,
                        )

                        # Train accuracy
                        _, _, _, A2_tr = forward_prop(W1, b1, W2, b2, X_train, activation=activation)
                        train_acc = get_accuracy(get_predictions(A2_tr), Y_train)

                        # Val accuracy
                        _, _, _, A2_v = forward_prop(W1, b1, W2, b2, X_val, activation=activation)
                        val_acc = get_accuracy(get_predictions(A2_v), Y_val)

                        print(f"  train={train_acc:.4f}  val={val_acc:.4f}")
                        results.append({
                            "activation":  activation,
                            "initializer": initializer,
                            "loss":        loss,
                            "optimizer":   optimizer,
                            "train_acc":   train_acc,
                            "val_acc":     val_acc,
                        })

                    except Exception as e:
                        print(f"  ERROR: {e}")
                        results.append({
                            "activation":  activation,
                            "initializer": initializer,
                            "loss":        loss,
                            "optimizer":   optimizer,
                            "train_acc":   float("nan"),
                            "val_acc":     float("nan"),
                        })

    return results


def format_table(results):
    """Format results as a markdown table string."""
    results_sorted = sorted(results, key=lambda r: r["val_acc"], reverse=True)

    header  = "| # | Activation | Initializer | Loss | Optimizer | Train Acc | Val Acc |"
    divider = "|---|------------|-------------|------|-----------|-----------|---------|"
    rows    = [header, divider]

    for rank, r in enumerate(results_sorted, 1):
        rows.append(
            f"| {rank} "
            f"| {r['activation']:10s} "
            f"| {r['initializer']:11s} "
            f"| {r['loss']:12s} "
            f"| {r['optimizer']:9s} "
            f"| {r['train_acc']:.4f}    "
            f"| {r['val_acc']:.4f}  |"
        )

    return "\n".join(rows)


def main():
    print(f"Loading data from {config.DATA_PATH} ...")
    data = load_data(config.DATA_PATH)
    X_train, Y_train, X_val, Y_val = split_data(data, val_size=config.VAL_SIZE)
    print(f"Train: {X_train.shape[1]} | Val: {X_val.shape[1]}")
    print(f"Grid: {EXPERIMENT_ITERATIONS} iterations, lr={ALPHA}, hidden={HIDDEN_SIZE}\n")

    results = run_experiments(X_train, Y_train, X_val, Y_val)

    table_str = format_table(results)

    print("\n\n── Results (sorted by Val Acc) ─────────────────────────────────────\n")
    print(table_str)

    # Save results to file
    output = (
        f"# Experiment Results\n\n"
        f"**Grid:** {EXPERIMENT_ITERATIONS} iterations · lr={ALPHA} · hidden={HIDDEN_SIZE}\n\n"
        + table_str + "\n"
    )
    with open("results.md", "w") as f:
        f.write(output)

    print(f"\nResults saved to results.md")


if __name__ == "__main__":
    main()
