import numpy as np

class Perceptron:

    def __init__(self, learning_rate=0.1, n_iter=100):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.w = None
        self.b = 0

    # Hàm huấn luyện
    def fit(self, X, y):

        # Khởi tạo trọng số bằng 0
        self.w = np.zeros(X.shape[1])

        # Lặp nhiều lần để huấn luyện
        for _ in range(self.n_iter):

            for i in range(len(X)):

                # Tính giá trị dự đoán
                z = np.dot(X[i], self.w) + self.b

                # Hàm bước
                y_pred = 1 if z >= 0 else -1

                # Nếu dự đoán sai thì cập nhật
                if y[i] != y_pred:
                    self.w = self.w + self.learning_rate * y[i] * X[i]
                    self.b = self.b + self.learning_rate * y[i]

        return self

    # Hàm dự đoán
    def predict(self, X):

        z = np.dot(X, self.w) + self.b

        return np.where(z >= 0, 1, -1)


# =========================
# DỮ LIỆU HUẤN LUYỆN
# =========================

X = np.array([
    [1, 1],
    [2, 1],
    [1, 2],
    [4, 4],
    [5, 4],
    [4, 5]
])

# Nhãn: -1 và +1
y = np.array([
    -1,
    -1,
    -1,
    1,
    1,
    1
])


# =========================
# HUẤN LUYỆN
# =========================

model = Perceptron(
    learning_rate=0.1,
    n_iter=100
)

model.fit(X, y)


# =========================
# DỰ ĐOÁN DỮ LIỆU MỚI
# =========================

X_new = np.array([
    [2, 2],
    [5, 5],
    [1, 3]
])

y_pred = model.predict(X_new)

print("Trọng số w:", model.w)
print("Bias b:", model.b)
print("Nhãn dự đoán:", y_pred)