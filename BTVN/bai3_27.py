import numpy as np
w = np.array([1, 2, -10])
x = np.array([3, 4, 1])
y_true = -1
w_dot_x = np.dot(w, x)
y_pred = 1 if w_dot_x >= 0 else -1
is_misclassified = (y_pred != y_true)
print(f"1. w^T * x = {w_dot_x}")
print(f"2. Nhãn dự đoán y_pred = {y_pred}")
print(f"3. Điểm dữ liệu có bị phân lớp sai không? {'Có' if is_misclassified else 'Không'}")