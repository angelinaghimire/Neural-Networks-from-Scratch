
## Features

- **4 Activations** — ReLU, Leaky ReLU, Tanh, Sigmoid
- **3 Initializers** — He (Kaiming), Xavier (Glorot), Naive
- **2 Loss Functions** — Cross-Entropy, MSE
- **4 Optimizers** — SGD, SGD + Momentum, RMSProp, Adam
- **Single config file** — change any option without touching source code
- **Experiment grid** — benchmark all combinations automatically

---


## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train a model
```bash
python3 trainingloop.py
```
Trains with the settings in `config.py` and saves weights to `model.npz`.

### 3. Evaluate and visualise predictions
```bash
python3 test.py
```
Loads `model.npz`, prints validation accuracy, and renders 3 random predictions as ASCII art.

### 4. Run the full experiment grid
```bash
python3 experiment.py
```
Trains every combination of activation × initializer × loss × optimizer and saves a ranked results table to `results.md`.

---

## Configuration

Edit [`config.py`](config.py) to control everything:

```python
# Hyperparameters
ITERATIONS   = 500
ALPHA        = 0.001       # learning rate
HIDDEN_SIZE  = 64

# Framework options
ACTIVATION   = "relu"             # 'relu' | 'leaky_relu' | 'tanh' | 'sigmoid'
INITIALIZER  = "he"               # 'naive' | 'he' | 'xavier'
LOSS         = "cross_entropy"    # 'cross_entropy' | 'mse'
OPTIMIZER    = "adam"             # 'sgd' | 'momentum' | 'rmsprop' | 'adam'
```

### Option Reference

| Setting | Options | Notes |
|---------|---------|-------|
| `ACTIVATION` | `relu` | Fast, default for most networks |
| | `leaky_relu` | Prevents dying ReLU neurons |
| | `tanh` | Output in (−1,1); pair with Xavier |
| | `sigmoid` | Classic, but slow to train; pair with Xavier |
| `INITIALIZER` | `he` | Best for ReLU / Leaky ReLU |
| | `xavier` | Best for Tanh / Sigmoid |
| | `naive` | Raw N(0,1); mostly for comparison |
| `LOSS` | `cross_entropy` | Standard for classification |
| | `mse` | Less stable with Softmax output |
| `OPTIMIZER` | `sgd` | Vanilla gradient descent; needs high lr |
| | `momentum` | SGD + exponential velocity; β=0.9 |
| | `rmsprop` | Adaptive lr per parameter; β=0.9 |
| | `adam` | Momentum + RMSProp; best default |

## Dataset

This project uses the [MNIST handwritten digit dataset](http://yann.lecun.com/exdb/mnist/) in CSV format (e.g. from [Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)).

- Place the training file at `data/train.csv`
- Format: first column is label (0–9), remaining 784 columns are pixel values (0–255)

---


## Experiment Results

64 combinations benchmarked · **200 iterations · lr=0.001 · hidden=64** · sorted by Val Acc.

| # | Activation | Initializer | Loss | Optimizer | Val Acc |
|---|-----------|------------|------|----------|---------|
| 1 | leaky_relu | he | cross_entropy | rmsprop | **94.10%** |
| 2 | relu | he | cross_entropy | adam | **93.80%** |
| 3 | tanh | xavier | mse | rmsprop | **93.80%** |
| 4 | relu | he | cross_entropy | rmsprop | 93.70% |
| 5 | tanh | xavier | cross_entropy | rmsprop | 93.40% |
| 6 | leaky_relu | he | cross_entropy | adam | 93.00% |
| 7 | relu | he | mse | adam | 92.90% |
| 8 | relu | he | mse | rmsprop | 92.40% |
| 9 | tanh | xavier | cross_entropy | adam | 92.40% |
| 10 | tanh | xavier | mse | adam | 92.30% |
| 11 | leaky_relu | he | mse | rmsprop | 92.20% |
| 12 | leaky_relu | he | mse | adam | 92.20% |
| 13 | sigmoid | xavier | cross_entropy | rmsprop | 90.30% |
| 14 | sigmoid | xavier | cross_entropy | adam | 87.20% |
| 15 | sigmoid | xavier | mse | rmsprop | 84.60% |
| ... | ... | ... | ... | ... | ... |
| 64 | tanh | naive | mse | momentum | 5.80% |

> Full table in [results.md](results.md). SGD and Momentum consistently underperform at lr=0.001 — they need a higher learning rate (≥0.1). Naive initialization hurts all activations significantly.

### Key Takeaways

| Finding | Detail |
|---------|--------|
|  Best combo | `leaky_relu + he + cross_entropy + rmsprop` → **94.1% val acc** at 200 iters |
|  Fastest convergence | RMSProp & Adam dominate; SGD needs ~10× higher lr |
|  Init matters | He/Xavier outperform Naive by **15–30%** in most combos |
| MSE surprise | Works well with Tanh+Xavier (93.8%), but bad with ReLU+Naive |
| Worst combos | Naive init + Momentum/SGD → near-random (5–17%) |

