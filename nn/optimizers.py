import numpy as np


class SGD:
    """
    Vanilla Stochastic Gradient Descent.

    Parameters
    ----------
    alpha : float
        Learning rate.
    """
    def __init__(self, alpha):
        self.alpha = alpha

    def step(self, W1, b1, W2, b2, dW1, db1, dW2, db2):
        W1 = W1 - self.alpha * dW1
        b1 = b1 - self.alpha * db1
        W2 = W2 - self.alpha * dW2
        b2 = b2 - self.alpha * db2
        return W1, b1, W2, b2


class SGDMomentum:
    """
    SGD with Momentum.

    Accumulates a velocity vector in the gradient direction,
    smoothing noisy gradients and accelerating convergence.

    Parameters
    ----------
    alpha : float
        Learning rate.
    beta : float
        Momentum coefficient (default 0.9).
    """
    def __init__(self, alpha, beta=0.9):
        self.alpha = alpha
        self.beta = beta
        self.vW1 = self.vb1 = self.vW2 = self.vb2 = None

    def step(self, W1, b1, W2, b2, dW1, db1, dW2, db2):
        # Initialise velocity on first call
        if self.vW1 is None:
            self.vW1 = np.zeros_like(W1)
            self.vb1 = np.zeros_like(b1)
            self.vW2 = np.zeros_like(W2)
            self.vb2 = np.zeros_like(b2)

        self.vW1 = self.beta * self.vW1 + (1 - self.beta) * dW1
        self.vb1 = self.beta * self.vb1 + (1 - self.beta) * db1
        self.vW2 = self.beta * self.vW2 + (1 - self.beta) * dW2
        self.vb2 = self.beta * self.vb2 + (1 - self.beta) * db2

        W1 = W1 - self.alpha * self.vW1
        b1 = b1 - self.alpha * self.vb1
        W2 = W2 - self.alpha * self.vW2
        b2 = b2 - self.alpha * self.vb2
        return W1, b1, W2, b2


class RMSProp:
    """
    RMSProp (Root Mean Square Propagation).

    Maintains a moving average of squared gradients and divides
    the learning rate by its square root, adapting per-parameter.

    Parameters
    ----------
    alpha : float
        Learning rate.
    beta : float
        Decay rate for the squared-gradient cache (default 0.9).
    eps : float
        Small constant to avoid division by zero (default 1e-8).
    """
    def __init__(self, alpha, beta=0.9, eps=1e-8):
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.sW1 = self.sb1 = self.sW2 = self.sb2 = None

    def step(self, W1, b1, W2, b2, dW1, db1, dW2, db2):
        if self.sW1 is None:
            self.sW1 = np.zeros_like(W1)
            self.sb1 = np.zeros_like(b1)
            self.sW2 = np.zeros_like(W2)
            self.sb2 = np.zeros_like(b2)

        self.sW1 = self.beta * self.sW1 + (1 - self.beta) * dW1 ** 2
        self.sb1 = self.beta * self.sb1 + (1 - self.beta) * db1 ** 2
        self.sW2 = self.beta * self.sW2 + (1 - self.beta) * dW2 ** 2
        self.sb2 = self.beta * self.sb2 + (1 - self.beta) * db2 ** 2

        W1 = W1 - self.alpha * dW1 / (np.sqrt(self.sW1) + self.eps)
        b1 = b1 - self.alpha * db1 / (np.sqrt(self.sb1) + self.eps)
        W2 = W2 - self.alpha * dW2 / (np.sqrt(self.sW2) + self.eps)
        b2 = b2 - self.alpha * db2 / (np.sqrt(self.sb2) + self.eps)
        return W1, b1, W2, b2


class Adam:
    """
    Adam (Adaptive Moment Estimation).

    Combines Momentum (first moment) and RMSProp (second moment)
    with bias correction. Generally the best default optimizer.

    Parameters
    ----------
    alpha : float
        Learning rate.
    beta1 : float
        First-moment decay (default 0.9).
    beta2 : float
        Second-moment decay (default 0.999).
    eps : float
        Numerical stability constant (default 1e-8).
    """
    def __init__(self, alpha, beta1=0.9, beta2=0.999, eps=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.mW1 = self.mb1 = self.mW2 = self.mb2 = None
        self.vW1 = self.vb1 = self.vW2 = self.vb2 = None

    def step(self, W1, b1, W2, b2, dW1, db1, dW2, db2):
        if self.mW1 is None:
            self.mW1 = np.zeros_like(W1); self.vW1 = np.zeros_like(W1)
            self.mb1 = np.zeros_like(b1); self.vb1 = np.zeros_like(b1)
            self.mW2 = np.zeros_like(W2); self.vW2 = np.zeros_like(W2)
            self.mb2 = np.zeros_like(b2); self.vb2 = np.zeros_like(b2)

        self.t += 1
        b1c = 1 - self.beta1 ** self.t   # bias-correction factor
        b2c = 1 - self.beta2 ** self.t

        for param, grad, m_attr, v_attr in [
            ('W1', dW1, 'mW1', 'vW1'), ('b1', db1, 'mb1', 'vb1'),
            ('W2', dW2, 'mW2', 'vW2'), ('b2', db2, 'mb2', 'vb2'),
        ]:
            m = getattr(self, m_attr)
            v = getattr(self, v_attr)
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * grad ** 2
            setattr(self, m_attr, m)
            setattr(self, v_attr, v)

        def _update(param, m_attr, v_attr):
            m_hat = getattr(self, m_attr) / b1c
            v_hat = getattr(self, v_attr) / b2c
            return param - self.alpha * m_hat / (np.sqrt(v_hat) + self.eps)

        W1 = _update(W1, 'mW1', 'vW1')
        b1 = _update(b1, 'mb1', 'vb1')
        W2 = _update(W2, 'mW2', 'vW2')
        b2 = _update(b2, 'mb2', 'vb2')
        return W1, b1, W2, b2


def get_optimizer(name, alpha):
    """
    Factory function — return an optimizer instance by name.

    Parameters
    ----------
    name  : str — 'sgd', 'momentum', 'rmsprop', or 'adam'
    alpha : float — learning rate

    Returns
    -------
    Optimizer instance with a .step() method.
    """
    name = name.lower()
    if name == 'momentum':
        return SGDMomentum(alpha)
    elif name == 'rmsprop':
        return RMSProp(alpha)
    elif name == 'adam':
        return Adam(alpha)
    else:
        return SGD(alpha)
