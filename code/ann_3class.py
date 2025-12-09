import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

# ----------------------------------------------------
# 1) Dataset (3 classes, 2 features)
# ----------------------------------------------------
N = 900
X = np.random.randn(N, 2)
y = np.random.randint(0, 3, size=N)

# train, validation, test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
X_val, X_test,  y_val,  y_test  = train_test_split(X_temp, y_temp, test_size=0.5)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_val   = torch.tensor(X_val,   dtype=torch.float32)
y_val   = torch.tensor(y_val,   dtype=torch.long)

X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_test  = torch.tensor(y_test,  dtype=torch.long)

# ----------------------------------------------------
# 2) ANN Model
# ----------------------------------------------------
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3)   # logits
        )

    def forward(self, x):
        return self.layers(x)

model = ANN()

# ----------------------------------------------------
# 3) Training setup
# ----------------------------------------------------
criterion = nn.CrossEntropyLoss()   # includes softmax internally
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ----------------------------------------------------
# 4) Training
# ----------------------------------------------------
epochs = 60

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    logits = model(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()

    # validation loss
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val), y_val)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Train: {loss.item():.4f} | Val: {val_loss.item():.4f}")

# ----------------------------------------------------
# 5) Getting validation predictions
# ----------------------------------------------------
softmax = nn.Softmax(dim=1)

model.eval()
with torch.no_grad():
    val_logits = model(X_val)
    val_probs = softmax(val_logits)        # probabilities
    val_pred_classes = val_probs.argmax(1) # predicted class indices

print("\nValidation predicted classes:\n", val_pred_classes)
print("\nValidation probabilities (first 5):\n", val_probs[:5])

# ----------------------------------------------------
# 6) Getting test predictions
# ----------------------------------------------------
with torch.no_grad():
    test_logits = model(X_test)
    test_probs = softmax(test_logits)
    test_pred_classes = test_probs.argmax(1)

print("\nTest predicted classes:\n", test_pred_classes)
print("\nTest probabilities (first 5):\n", test_probs[:5])

# accuracy examples
val_acc  = (val_pred_classes  == y_val).float().mean().item()
test_acc = (test_pred_classes == y_test).float().mean().item()

print(f"\nValidation accuracy: {val_acc:.3f}")
print(f"Test accuracy:       {test_acc:.3f}")

