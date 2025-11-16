import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
np.random.seed(0)

X = np.random.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

n, d = X.shape


# ---------------------------------------------------------
# 1. Logistic Regression (manual implementation)
# ---------------------------------------------------------
def train_logistic_regression(X, y, lr=0.1, epochs=2000):
    n, d = X.shape
    W = np.zeros((d, 1))
    b = 0.0

    for _ in range(epochs):
        z = X @ W + b
        p = sigmoid(z)

        dW = (1/n) * X.T @ (p - y)
        db = (1/n) * np.sum(p - y)

        W -= lr * dW
        b -= lr * db

    return W, b


W_lr, b_lr = train_logistic_regression(X, y)
pred_lr = sigmoid(X @ W_lr + b_lr)


# ---------------------------------------------------------
# 2. Single Neuron ANN (manual implementation)
# ---------------------------------------------------------
def train_single_neuron(X, y, lr=0.1, epochs=2000):
    n, d = X.shape
    W = np.zeros((d, 1))
    b = 0.0

    for _ in range(epochs):
        z = X @ W + b
        a = sigmoid(z)

        dW = (1/n) * X.T @ (a - y)
        db = (1/n) * np.sum(a - y)

        W -= lr * dW
        b -= lr * db

    return W, b


W_nn, b_nn = train_single_neuron(X, y)
pred_nn = sigmoid(X @ W_nn + b_nn)


# ---------------------------------------------------------
# 3. PyTorch Single Neuron Model
# ---------------------------------------------------------
class SingleNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


torch.manual_seed(0)
model = SingleNeuron()

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)

for _ in range(2000):
    optimizer.zero_grad()
    pred = model(X_t)
    loss = criterion(pred, y_t)
    loss.backward()
    optimizer.step()

pred_torch = model(X_t).detach().numpy()

# Extract torch weights
W_t = model.linear.weight.detach().numpy().reshape(-1, 1)
b_t = model.linear.bias.detach().numpy()[0]


# ---------------------------------------------------------
# Comparison
# ---------------------------------------------------------
print("\n--- Weight Comparison ---")
print("Manual Logistic Regression W:", W_lr.ravel())
print("Manual Neuron W:             ", W_nn.ravel())
print("PyTorch W:                   ", W_t.ravel())

print("\n--- Bias Comparison ---")
print("Logistic Regression b:", b_lr)
print("Manual Neuron b:      ", b_nn)
print("PyTorch b:            ", b_t)

print("\n--- Prediction Differences (mean absolute error) ---")
print("LR vs Manual NN:   ", np.mean(np.abs(pred_lr - pred_nn)))
print("LR vs PyTorch NN:  ", np.mean(np.abs(pred_lr - pred_torch)))
print("NN vs PyTorch NN:  ", np.mean(np.abs(pred_nn - pred_torch)))

print("\nDone!")

