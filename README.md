# 🧠 Neural Network from Scratch

A lightweight, modular **2-layer Neural Network** built entirely from scratch using only Python and NumPy. This project is designed to classify the classic MNIST digit dataset (28x28 grayscale images of handwritten digits 0-9) and serves as an excellent educational baseline for understanding how Deep Learning works under the hood.

---

## Features

- **Built from Scratch**: No PyTorch, TensorFlow, or Keras. Every matrix multiplication and derivative is calculated manually.
- **Modular Design**: Code is cleanly separated into specialized modules (`forward.py`, `backward.py`, etc.) for easy experimentation.

## Getting Started

### 1. Prerequisites
You only need two standard libraries to run this project:
```bash
pip install numpy pandas
```
*(Note: A dataset file like `data/train.csv` containing the MNIST digits is required).*

### 2. Configuration
Open `config.py` to tweak your hyperparameters before training:
```python
DATA_PATH    = "data/train.csv"
ITERATIONS   = 500                
ALPHA        = 0.10               # Learning rate
```

### 3. Training the Model
Run the training loop to parse the data, execute gradient descent, and save the learned weights to `model.npz`:
```bash
python3 trainingloop.py
```

### 4. Testing & Visualizing
Once the model is trained, run the test script. It will load the saved weights and predict random digits from the validation set, rendering them as ASCII art right in your terminal:
```bash
python3 test.py
```

---

## 🔬 The Math Under the Hood

This is a standard 2-layer perceptron:
1. **Input Layer**: 784 neurons (28x28 flattened image pixels).
2. **Hidden Layer**: 10 neurons with **ReLU** activation.
3. **Output Layer**: 10 neurons (representing classes 0-9) with **Softmax** activation.
4. **Optimization**: Standard Gradient Descent using **Categorical Cross-Entropy** (implicitly calculated during backprop via $dZ^{[2]} = A^{[2]} - Y$).

---
