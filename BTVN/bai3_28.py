import numpy as np
w = np.array([-2, 1, 0])
x = np.array([2, 3, 1])
y = 1

def perceptron_update(w, x, y, eta = 1):
    if y * np.dot(w, x) <= 0:
        w += y * x * eta
    return w

wTx = np.dot(w, x)
print(f"1. w^T * x = {wTx}")
# wTx = -1

y_pred1 = 1 if wTx >= 0 else -1
print(f"2. Predicted y: {y_pred1} | True y: {y}")
# y_pred = sgn(wTx) = -1

is_misclassified = (y_pred1 != y)
print(f"3. Is misclassified?: {is_misclassified}")
# y_true = 1 != y_pred
# ==> Misclassified

w_updated = perceptron_update(w, x, y)
print(f"4. New w: {w_updated}")
# Update w: w_new = w + eta*y*x = (0, 4, 1)

wTx_updated = np.dot(w_updated, x)
print(f"5. New w^T * x = {wTx_updated}")