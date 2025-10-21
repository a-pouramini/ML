import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# داده واقعی با نویز ذاتی
np.random.seed(0)
X = np.linspace(-1, 1, 20).reshape(-1, 1)
y = 2 * X.flatten() + 1 + np.random.normal(0, 0.5, size=X.shape[0])

# آموزش مدل رگرسیون خطی
model = LinearRegression().fit(X, y)

# پیش‌بینی در بازه‌ی بزرگ‌تر
X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
y_pred = model.predict(X_test)

plt.scatter(X, y, label="Train data")
plt.plot(X_test, y_pred, color="r", label="Model prediction")
plt.fill_between(X_test.flatten(), y_pred - 0.5, y_pred + 0.5,
                 color="orange", alpha=0.3, label="Aleatoric uncertainty")
plt.title("Aleatoric vs Epistemic Uncertainty (conceptually)")
plt.legend()
plt.show()

