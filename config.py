DATA_PATH    = "data/train.csv"   # path to the CSV dataset
VAL_SIZE     = 1000               # number of examples held out for validation
ITERATIONS   = 500                # training iterations
ALPHA        = 0.001              # learning rate (Adam default; use 0.1 for SGD)
MODEL_PATH   = "model.npz"        # where trained weights are saved

# ── Network Architecture ─────────────────────────────────────────────────────
INPUT_SIZE   = 784                # 28×28 pixel images
HIDDEN_SIZE  = 64                 # hidden layer width
OUTPUT_SIZE  = 10                 # digit classes (0-9)

# ── Framework Options ────────────────────────────────────────────────────────
ACTIVATION   = "relu"             # 'relu' | 'leaky_relu' | 'tanh' | 'sigmoid'
INITIALIZER  = "he"               # 'naive' | 'he' | 'xavier'
LOSS         = "cross_entropy"    # 'cross_entropy' | 'mse'
OPTIMIZER    = "adam"             # 'sgd' | 'momentum' | 'rmsprop' | 'adam'
